import ml_method as ml
import dl_methods as dl
import argparse


def main(config): 
    ml.main_processing(config['data'],config['balancing'])     
    # dl.DNN_training()
    #dl.MLP()
    # dl.Conv1D_training()
    # dl.RNN_training()
    dl.deep_learning_method(config['data'],config['balancing'])
    
  
# __name__ 
if __name__=="__main__": 
    parser = argparse.ArgumentParser(description="Asteroid Classification")
    parser.add_argument("-b",'--balancing',help = "balancing_method",required=False)
    parser.add_argument("-d",'--data',help="Data",required = True)
    args = parser.parse_args()
    config = vars(args)
    if 'neo' in config['data'].lower():
        config['data'] = "NeoWs"
    else:
        config['data'] = "Asteroids"
    if config['balancing']:
        if 'smo' in config['balancing'].lower():
            config['balancing'] = "smote"
        else:
            config['balancing'] = "bootstrapping"
    else:
        config['balancing'] = "bootstrapping"
    print(config['data'],config['balancing'])
    main(config) 
