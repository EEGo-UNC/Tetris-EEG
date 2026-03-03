import numpy as np

R_MAX = 2.828 
PHI_SCALE = np.pi
FLOW_ANGLE = np.pi / 4


def delta_s_vec(valence, arousal, activation_constant=1, centerX=2.6, centerY=2.6):
    """Calculate a normalized speed adjustment amplitude 'delta_s' based on valence and arousal.
    Args:
        valence (float or np.ndarray): Valence value(s) Range: [1, 5].
        arousal (float or np.ndarray): Arousal value(s) Range: [1, 5].
        activation_constant (float): Constant to control the steepness of the envelope. Default is 1
        centerX (float): Center x-coordinate. Default is 3.0
        centerY (float): Center y-coordinate. Default is 3.0
    Returns:
        float or np.ndarray: The speed adjustment amplitude 'delta_s'. In a range approximately [-1, 1].
    """
    if np.isnan(valence) or np.isnan(arousal):
        return 0

    #Centered coordinates to center at 3,3
    x = valence - centerX
    y = arousal - centerY
    
    # Find the polar coordinates of valence and arousal
    r = np.hypot(x, y)
    theta = np.arctan2(y, x)

    # Target flow angle is pi/4

    phi = theta - FLOW_ANGLE
    
    # Angular component, normalizing to remove pi dependence
    s = phi / PHI_SCALE

    # A regularizing envelope that tapers off to 0 at the center and edges
    env = np.tanh(activation_constant * r) * (r / R_MAX)

    # Negate to ensure that positive delta_s corresponds to movement away from the flow angle (pi/4),
    # so that the direction of adjustment matches the intended flow direction in the valence-arousal space.
    return -1 * env * s
