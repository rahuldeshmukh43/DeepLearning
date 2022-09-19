import os

import config
from src.NCN.model import NCNetModel
from data.pf_pascal_dataset import PF_PASCAL_DATALOADER  

# we test always with batch size of one
def test_pf_pascal(args):
    # save a copy for the current args in out_folder
    out_folder = os.path.join(args.outdir, args.exp_name, 'testing_results')
    os.makedirs(out_folder, exist_ok=True)
    f = os.path.join(out_folder, 'args.txt')
    with open(f, 'w') as file:
        for arg in vars(args):
            attr = getattr(args, arg)
            file.write('{} = {}\n'.format(arg, attr))
    
    # pf pascal data loader
    test_loader = PF_PASCAL_DATALOADER(args).load_data()
    
    #define model
    model = NCNetModel(args)
    start_step = model.start_step 
    for data in test_loader:
        model.set_input(data)
        model.test()
        matches, _matching_scores = model.get_matches_and_matching_score() #wont work as matches not computed, loss computes that
        eval_score = model.compute_eval_score(matches)
        
        #draw top matches
        
    
    return




if __name__ == "__main__":
    args = config.get_args()
    test_pf_pascal(args)