import ml_method as ml
import dl_methods as dl
import argparse


def main(config): 
    ml.main_processing(config['data'],config['balancing'])     
    #dl.DNN_training(config['data'],config['balancing'])
    #dl.MLP(config['data'],config['balancing'])
    #dl.Conv3D_training(config['data'],config['balancing'])
    dl.RNN_training()
    
  
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
