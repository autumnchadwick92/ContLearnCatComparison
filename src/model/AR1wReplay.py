from __future__ import print_function
from __future__ import division
from __future__ import absolute_import

#from data_loader import CORE50
import copy
import os
import json

from torch.utils.data import DataLoader

from src.model.AR1Model.mobilenet import MyMobilenetV1
from src.model.AR1Model.utils import *
import configparser
import argparse
from pprint import pprint
from torch.utils.tensorboard import SummaryWriter

# --------------------------------- Setup --------------------------------------



def AR1wReplay(exp_name,tasks_data,test_data,batch_size):

    # set cuda device (based on your hardware)
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"

    exp_name = "ar1-free_v1"
    comment = "with latent replay"
    use_cuda = False
    init_lr = 0.001
    inc_lr = 0.00005
    mb_size = 128
    init_train_ep = 4
    inc_train_ep = 4
    init_update_rate = 0.01
    inc_update_rate = 0.00005
    max_r_max = 1.25
    max_d_max = 0.5
    inc_step = 4.1e-05
    rm_sz = 1500
    momentum = 0.9
    l2 = 0.0005
    freeze_below_layer = "lat_features.19.bn.beta"
    latent_layer_num = 19
    reg_lambda = 0
    rm = None


    # setting up log dir for tensorboard
    log_dir = 'logs/' + exp_name
    writer = SummaryWriter(log_dir)

    # Saving params
    #hyper = json.dumps(dict(exp_config))
    #writer.add_text("parameters", hyper, 0)

    # Other variables init
    tot_it_step = 0
    rm = None

    # Create the dataset object
    #dataset = CORE50(root='/home/admin/ssd_data/core50', scenario="nicv2_391")
    preproc = preprocess_imgs


    # Model setup
    model = MyMobilenetV1(pretrained=True, latent_layer_num=latent_layer_num)
    # we replace BN layers with Batch Renormalization layers
    replace_bn_with_brn(
        model, momentum=init_update_rate, r_d_max_inc_step=inc_step,
        max_r_max=max_r_max, max_d_max=max_d_max
    )
    model.saved_weights = {}
    model.past_j = {i:0 for i in range(50)}
    model.cur_j = {i:0 for i in range(50)}
    if reg_lambda != 0:
        # the regularization is based on Synaptic Intelligence as described in the
        # paper. ewcData is a list of two elements (best parametes, importance)
        # while synData is a dictionary with all the trajectory data needed by SI
        ewcData, synData = create_syn_data(model)

    # Optimizer setup
    optimizer = torch.optim.SGD(
        model.parameters(), lr=init_lr, momentum=momentum, weight_decay=l2
    )
    criterion = torch.nn.CrossEntropyLoss()

    # --------------------------------- Training -----------------------------------
    for task_id, (train_set) in enumerate(tasks_data):
            # loop over the training incremental batches
        dataset = DataLoader(train_set, batch_size=batch_size, shuffle=True)

        for i, train_batch in enumerate(dataset):

            if reg_lambda != 0:
                init_batch(model, ewcData, synData)

            # we freeze the layer below the replay layer since the first batch
            freeze_up_to(model, freeze_below_layer, only_conv=False)

            if i == 1:
                change_brn_pars(
                    model, momentum=inc_update_rate, r_d_max_inc_step=0,
                    r_max=max_r_max, d_max=max_d_max)
                optimizer = torch.optim.SGD(
                    model.parameters(), lr=inc_lr, momentum=momentum, weight_decay=l2
                )

            train_x, train_y, task = train_batch
            #train_x = preproc(train_x)

            if i == 0:
                cur_class = [int(o) for o in set(train_y)]
                model.cur_j = examples_per_class(train_y)
            else:
                cur_class = [int(o) for o in set(train_y).union(set(rm[1]))]
                model.cur_j = examples_per_class(list(train_y) + list(rm[1]))

            print("----------- batch {0} -------------".format(i))
            print("train_x shape: {}, train_y shape: {}"
                  .format(train_x.shape, train_y.shape))

            model.train()
            model.lat_features.eval()

            reset_weights(model, cur_class)
            cur_ep = 0

            if i == 0:
                (train_x, train_y), it_x_ep = pad_data([train_x, train_y], mb_size)
            #shuffle_in_unison([train_x, train_y], in_place=True)
            #print(train_x)
            model = maybe_cuda(model, use_cuda=use_cuda)
            acc = None
            ave_loss = 0

            if not(torch.is_tensor(train_x)):
                train_x = torch.from_numpy(train_x).type(torch.FloatTensor)
                train_y = torch.from_numpy(train_y).type(torch.LongTensor)

            if i == 0:
                train_ep = init_train_ep
            else:
                train_ep = inc_train_ep

            for ep in range(train_ep):

                print("training ep: ", ep)
                correct_cnt, ave_loss = 0, 0

                # computing how many patterns to inject in the latent replay layer
                if i > 0:
                    cur_sz = train_x.size(0) // ((train_x.size(0) + rm_sz) // mb_size)
                    it_x_ep = train_x.size(0) // cur_sz
                    n2inject = max(0, mb_size - cur_sz)
                else:
                    n2inject = 0
                print("total sz:", train_x.size(0) + rm_sz)
                print("n2inject", n2inject)
                print("it x ep: ", it_x_ep)

                for it in range(it_x_ep):

                    if reg_lambda !=0:
                        pre_update(model, synData)

                    start = it * (mb_size - n2inject)
                    end = (it + 1) * (mb_size - n2inject)

                    optimizer.zero_grad()

                    x_mb = maybe_cuda(train_x[start:end], use_cuda=use_cuda)

                    if i == 0:
                        lat_mb_x = None
                        y_mb = maybe_cuda(train_y[start:end], use_cuda=use_cuda)

                    else:
                        lat_mb_x = rm[0][it*n2inject: (it + 1)*n2inject]
                        lat_mb_y = rm[1][it*n2inject: (it + 1)*n2inject]
                        y_mb = maybe_cuda(
                            torch.cat((train_y[start:end], lat_mb_y), 0),
                            use_cuda=use_cuda)
                        lat_mb_x = maybe_cuda(lat_mb_x, use_cuda=use_cuda)

                    # if lat_mb_x is not None, this tensor will be concatenated in
                    # the forward pass on-the-fly in the latent replay layer
                    logits, lat_acts = model(
                        x_mb, latent_input=lat_mb_x, return_lat_acts=True)

                    # collect latent volumes only for the first ep
                    # we need to store them to eventually add them into the external
                    # replay memory
                    if ep == 0:
                        lat_acts = lat_acts.cpu().detach()
                        if it == 0:
                            cur_acts = copy.deepcopy(lat_acts)
                        else:
                            cur_acts = torch.cat((cur_acts, lat_acts), 0)

                    _, pred_label = torch.max(logits, 1)
                    correct_cnt += (pred_label == y_mb).sum()

                    loss = criterion(logits, y_mb)
                    if reg_lambda !=0:
                        loss += compute_ewc_loss(model, ewcData, lambd=reg_lambda)
                    ave_loss += loss.item()

                    loss.backward()
                    optimizer.step()

                    if reg_lambda !=0:
                        post_update(model, synData)

                    acc = correct_cnt.item() / \
                          ((it + 1) * y_mb.size(0))
                    ave_loss /= ((it + 1) * y_mb.size(0))

                    if it % 10 == 0:
                        print(
                            '==>>> it: {}, avg. loss: {:.6f}, '
                            'running train acc: {:.3f}'
                                .format(it, ave_loss, acc)
                        )

                    # Log scalar values (scalar summary) to TB
                    tot_it_step +=1
                    writer.add_scalar('train_loss', ave_loss, tot_it_step)
                    writer.add_scalar('train_accuracy', acc, tot_it_step)

                cur_ep += 1

            consolidate_weights(model, cur_class)
            if reg_lambda != 0:
                update_ewc_data(model, ewcData, synData, 0.001, 1)

            # how many patterns to save for next iter
            h = min(rm_sz // (i + 1), cur_acts.size(0))
            print("h", h)

            print("cur_acts sz:", cur_acts.size(0))
            idxs_cur = np.random.choice(
                cur_acts.size(0), h, replace=False
            )
            rm_add = [cur_acts[idxs_cur], train_y[idxs_cur]]
            print("rm_add size", rm_add[0].size(0))

            # replace patterns in random memory
            if i == 0:
                rm = copy.deepcopy(rm_add)
            else:
                idxs_2_replace = np.random.choice(
                    rm[0].size(0), h, replace=False
                )
                for j, idx in enumerate(idxs_2_replace):
                    rm[0][idx] = copy.deepcopy(rm_add[0][j])
                    rm[1][idx] = copy.deepcopy(rm_add[1][j])

            set_consolidate_weights(model)

            """
            for test_id, (test_set) in enumerate(tasks_data):
                # Get the fixed test set
                test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False)
                for i, test_batch in enumerate(test_loader):

                    test_x, test_y, task = test_batch
                    print(test_x)

                    ave_loss, acc, accs = get_accuracy(
                        model, criterion, mb_size, test_x, test_y
                    )

                    # Log scalar values (scalar summary) to TB
                    writer.add_scalar('test_loss', ave_loss, i)
                    writer.add_scalar('test_accuracy', acc, i)

                    # update number examples encountered over time
                    for c, n in model.cur_j.items():
                        model.past_j[c] += n

                    print("---------------------------------")
                    print("Accuracy: ", acc)
                    print("---------------------------------")
            """
    writer.close()