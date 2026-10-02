import numpy as np
import math

FEATURE_VERSION = 1

def canonicalize(landmarks: np.ndarray, handedness: str = 'Right', 
                 mirror_to_right: bool = True,
                 normalize_rotation: bool = True,
                 z_weight: float = 0.3,
                 scale_epsilon: float = 1e-6) -> np.ndarray:
    """Canonicalize hand landmarks for static-pose classification.
    
    Steps:
    1. Wrist-origin: translate so wrist (index 0) is at origin
    2. Scale normalization: divide by max distance from wrist, with epsilon
    3. Mirror handedness: if left hand and mirror_to_right, flip x-axis
    4. Rotation normalization: rotate so middle finger MCP (index 9) 
       is along positive y-axis in the xy-plane (palm-axis alignment)
    5. Z-axis weighting: multiply z by z_weight (z is less reliable from single camera)
    
    Limitations of z normalization:
    - Single camera depth estimation is noisy
    - z_weight < 1.0 reduces z influence but doesn't eliminate noise
    - True 3D normalization would require stereo/depth camera
    
    Returns: flattened array of shape (63,) = 21 landmarks * 3 coords
    """
    if landmarks.shape != (21, 3):
        raise ValueError("Landmarks must have shape (21, 3)")
        
    out = landmarks.copy().astype(float)
    
    # 1. Wrist-origin
    wrist = out[0].copy()
    out -= wrist
    
    # 2. Scale normalization
    max_dist = np.max(np.linalg.norm(out, axis=1))
    scale = max(max_dist, scale_epsilon)
    out /= scale
    
    # 3. Mirror handedness
    if handedness.lower() == 'left' and mirror_to_right:
        out[:, 0] = -out[:, 0]
        
    # 4. Rotation normalization
    if normalize_rotation:
        # Middle finger MCP is at index 9
        mcp = out[9]
        # Angle of vector in xy-plane
        # We want to rotate so that mcp is along +y axis
        # Current angle theta
        theta = math.atan2(mcp[1], mcp[0])
        # We want final angle to be pi/2
        # rotation angle alpha = pi/2 - theta
        alpha = math.pi / 2 - theta
        
        c, s = math.cos(alpha), math.sin(alpha)
        # 2D rotation matrix
        # [x', y'] = [x*c - y*s, x*s + y*c]
        x = out[:, 0].copy()
        y = out[:, 1].copy()
        
        out[:, 0] = x * c - y * s
        out[:, 1] = x * s + y * c
        
    # 5. Z-axis weighting
    out[:, 2] *= z_weight
    
    return out.flatten()
