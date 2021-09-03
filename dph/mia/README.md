# Membership Inference Attack

## Structure

1. `mia_experiment.py`:
    Represents a Membership Inference Attack experiment.
    Can be configured with `attack_parameters.py`

2. `features`: 

    From a Target model we need to get
    - Gradient matrix w.r.t. to the inputs
    - Layer Outputs
    - Loss
    - Label
    - (Attention weights)

    i. `extract`
    
    Classes needed to load target models and extract their features.
    
    ii. `serialize`
    
    Classes needed to save/load attack features as `tf.data.Dataset`
    to/from `tfrecord` files.  
    
    iii. `analyze`

3. `attacker`

    i. The attacker uses the `features` files and an `attack_model`  to
    conduct membership inference attacks.
    
    ii. `attack_model`:  
    The Attack Model is a `tf.keras.Model`. 
    It is based upon the paper from Nasr et al. 
    regarding White-Box Membership Inference Attacks.