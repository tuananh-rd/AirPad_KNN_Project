from enum import IntEnum, auto

class HandPose(IntEnum):
    """Static hand poses classified by the configured pose classifier.
    These are STATIC poses only - no dynamic gestures here.
    """
    UNKNOWN = 0
    OPEN_PALM = auto()             # All fingers extended
    POINT_ONE = auto()             # Index finger extended, others closed
    PINCH_CLOSED = auto()          # Thumb-index pinch (static closed position)
    PINCH_MIDDLE = auto()          # Thumb-middle pinch (static pose for right click)
    TWO_FINGERS_PARALLEL = auto() # Index+middle extended parallel
    L_SHAPE = auto()               # Thumb+index in L shape
    THREE_FINGERS = auto()         # Index+middle+ring extended  
    FIST = auto()                  # All fingers closed
    OTHER = auto()                 # Catch-all
    NO_HAND = auto()               # No hand detected
