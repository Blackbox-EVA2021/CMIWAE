from glob import glob
from parse import parse
from tqdm import tqdm


def parse_main_output(fname):
    with open(fname) as file:
        lines = file.readlines()
        
    lines.reverse()
    
    valid_years = None
    for line in lines:
        res = parse("Minimum score: [{}], sum: {valid_score:g}\n", line)
        if res is not None:
            assert valid_years is not None
            return({'fname' : fname,
                    'valid_score' : res.named['valid_score'],
                    'valid_years' : valid_years
                   })
            
        res = parse("Validation set years: {years}", line)
        if res is not None:
            valid_years=eval(res.named['years'])
    
    return None

def parse_losses(fname):
    with open(fname) as file:
        lines = file.readlines()
    
        stats = []
        epoch = 0
        for line in lines:

            res = parse("{epoch:d} {}", line)
            if res is not None:
                assert res.named['epoch'] == epoch
                res2 = parse("{epoch:d}{:s}{train_loss:g}{:s}{valid_loss:g}{:s}{valid_CNT_score:g}{:s}{valid_BA_score:g}{:s}{valid_total_score:g}{:s}{}", line)
                if res2 is not None:
                    stats.append(res2.named)
                else:
                    return None
                
                epoch += 1
                
    return stats

def parse_all_outputs(fpath):
    main_stats = []
    for fname in glob(fpath):
        main_stats.append(parse_main_output(fname))
    
    print(f"  Total models (files) computed: {len(main_stats)}")
    main_stats = [r for r in main_stats if r is not None]
    print(f"  Total models left after purging models with unfinished training: {len(main_stats)}")
    
    train_stats = []
    retrain_models = []
    for r in tqdm(main_stats):
        res = parse_losses(r['fname'])
        if res is not None:
            train_stats.append({'fname' : r['fname'], 'train_stats' : res})
        else:
            retrain_models.append(r['fname'])
    
    return main_stats, train_stats, retrain_models

def validate_train_stats(train_stats, threshold=1, factor=10):
    for epoch_old, epoch_new in zip(train_stats[:-1], train_stats[1:]):
        delta_train_loss, delta_valid_loss = max(epoch_new['train_loss'] - epoch_old['train_loss'], 0), max(epoch_new['valid_loss'] - epoch_old['valid_loss'], 0)
        if delta_train_loss > max(epoch_old['train_loss'], threshold) or delta_valid_loss > max(factor * epoch_old['valid_loss'], threshold):
            print(delta_train_loss, delta_valid_loss)
            print(epoch_old, epoch_new)
            return epoch_old, epoch_new
    return None

def validate_all_models(fpath="outputs/output-run*-idx*.txt"):
    main_stats, train_stats, retrain_models_1 = parse_all_outputs(fpath)
    
    retrain_models = []
    for run in train_stats:
        res = validate_train_stats(run['train_stats'])
        if res is not None:
            retrain_models.append(run['fname'])
    
    return main_stats, train_stats, (retrain_models_1, retrain_models)

def parse_model_info(fn):
    res = parse("outputs/output-run{run:d}-idx{seed:d}.txt", fn)
    if res is not None:
        return res.named
    else:
        raise ValueError("Not a valid model output file name.")

def get_all_models(fpath="outputs/output-run*-idx*.txt", get_model_name_fcn=None, **kwargs):
    stats, _, retrain = validate_all_models(fpath)
    
    if(len(retrain[0] + retrain[1]) > 0):
        print("Retrain models:")
        print(retrain[0])
        print(retrain[1])
        print("---------------")
    
    stats = [stat for stat in stats if stat['fname'] not in (retrain[0] + retrain[1])]
    output_fnames =[s['fname'] for s in stats]
    
    if get_model_name_fcn is not None:
        models = [get_model_name_fcn(parse_model_info(fn), **kwargs) for fn in output_fnames]
    else:
        models = [parse_model_info(fn) for fn in output_fnames]
    idxs = [parse_model_info(fn)['seed'] for fn in output_fnames]
    
    return models, idxs