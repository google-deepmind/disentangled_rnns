import numpy as np

def fit_linear(update_dict):
    for latent_dict in update_dict.keys():
        for observation in update_dict[latent_dict].keys():
            coefficients = np.polyfit(
                update_dict[latent_dict][observation]['state_bins'], 
                update_dict[latent_dict][observation]['delta_states'], 
                1
                )
            update_dict[latent_dict][observation]['slope'] = coefficients[0]
            update_dict[latent_dict][observation]['intercept'] = coefficients[1]
    return update_dict

def print_linear(update_dict):
    for latent in update_dict.keys():
        print(latent)
        for obs in update_dict[latent].keys():
            slope = update_dict[latent][obs]['slope'][0]
            intercept = update_dict[latent][obs]['intercept'][0]
            print(obs)
            print('y{}'.format(latent)+ ' = ' + 'y{}'.format(latent) + ' + ' + '{:.2f}y{}'.format(slope,latent) + ' + ' + '{:.2f}'.format(intercept))
            print('fixed point: {:.3f}'.format(-intercept/slope))
            print()
        print("\n\n") 
