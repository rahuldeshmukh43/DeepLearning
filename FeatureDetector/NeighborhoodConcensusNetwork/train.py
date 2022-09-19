import os
from tensorboardX import SummaryWriter

import config
from src.NCN.model import NCNetModel
from data.pf_pascal_dataset import PF_PASCAL_DATALOADER
from utils import Setup_Logger
# from utils import cycle 

def train_pf_pascal(args):
    # save a copy for the current args in out_folder
    out_folder = os.path.join(args.outdir, args.exp_name)
    os.makedirs(out_folder, exist_ok=True)
    f = os.path.join(out_folder, 'args.txt')
    with open(f, 'w') as file:
        for arg in vars(args):
            attr = getattr(args, arg)
            file.write('{} = {}\n'.format(arg, attr))

    # tensorboard writer
    tb_log_dir = os.path.join(args.logdir, args.exp_name)
    print('tensorboard log files are stored in {}'.format(tb_log_dir))
    writer = SummaryWriter(tb_log_dir)

    #logger
    log_file = os.path.join(out_folder, args.log_name + '.log')
    logger = Setup_Logger(args.log_name , log_file )

    # pf pascal data loader
    train_loader = PF_PASCAL_DATALOADER(args).load_data()
    args.phase = 'val'
    validation_loader = PF_PASCAL_DATALOADER(args).load_data()
    #train_loader_iterator = iter(cycle(train_loader))

    # define model
    model = NCNetModel(args, logger)
    step = model.start_step

    # training loop
    # for step in range(start_step + 1, start_step + args.n_iters + 1):
    #     data = next(train_loader_iterator)
    #     model.set_input(data)
    #     model.optimize_parameters()
    #     model.write_summary(writer, step)
    #     if step % args.save_interval == 0 and step > 0:
    #         model.save_model(step)
    
    for epoch in range(args.num_epochs):
        #train
        running_train_loss=0.0
        for i_minibatch, data in enumerate(train_loader):  
            step += 1
            model.set_input(data)
            model.optimize_parameters()            
            model.write_train_summary(writer, step, epoch, i_minibatch)
            if step % args.save_interval == 0 and step > 0:
                model.save_model(step)
            if args.debug:
                break

        #eval validation
        eval_step=0
        running_vloss=0.0
        running_eval_score = 0.0
        for data in validation_loader:
            model.set_input(data)
            model.test()
            model.compute_loss()
            matches, _matching_scores = model.get_matches_and_matching_score()
            eval_score = model.compute_eval_score(matches)
            
            eval_loss = model.get_loss()
            running_vloss += eval_loss
            running_eval_score += eval_score
            
            model.write_eval_summary(writer, eval_step, epoch)
            eval_step+=1
            if args.debug:
                break
        
        #per epoch stats    
        avg_loss = running_train_loss/(i_minibatch + 1)
        avg_vloss  = running_vloss/(eval_step)
        avg_eval_score = running_eval_score/(eval_step)
        model.write_epoch_summary( writer, epoch, avg_loss, avg_vloss, avg_eval_score)
        
        #save model after every epoch
        model.save_model(step)
        if args.debug:
            break
    return
    

if __name__ == "__main__":
    args = config.get_args()
    train_pf_pascal(args)