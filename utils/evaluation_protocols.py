import numpy as np
from .metrics import compute_occ_3d_errors_batch, compute_3d_errors_batch, compute_acceleration_error


def _filter_hand_failure_cases(HAND_CS_MPJPE, HAND_ACC_ERROR, ignore_failure_solves):
    failure_cases = HAND_CS_MPJPE > 1000

    if ignore_failure_solves:
        HAND_CS_MPJPE = HAND_CS_MPJPE[~failure_cases]
        HAND_ACC_ERROR = HAND_ACC_ERROR[~failure_cases]

    FAILURE_RATE = np.mean(failure_cases.astype(np.float32)) * 100.0

    if HAND_CS_MPJPE.shape[0] == 0:
        HAND_CS_MPJPE = np.array([0.0])
        HAND_ACC_ERROR = np.array([0.0])

    return HAND_CS_MPJPE, HAND_ACC_ERROR, FAILURE_RATE


def evaluate_batch_hand_arm(data, ignore_failure_solves=False):
    pred_hand_j3d = data['pred_hand_j3d'] * 1000 # to mm
    pred_arm_j3d = data['pred_arm_j3d'] * 1000 # to mm

    gt_hand_j3d = data['gt_hand_j3d'] * 1000 # to mm
    gt_arm_j3d = data['gt_arm_j3d'] * 1000 # to mm

    visible_hand = np.array(data['visible_hand'], dtype=bool)
    valid_hand = np.array(data['valid_hand_j3d'], dtype=bool)

    visible_hand = np.logical_and(visible_hand, valid_hand)

    visible_arm = np.array(data['visible_arm'], dtype=bool)
    valid_arm = np.array(data['valid_arm_j3d'], dtype=bool)

    visible_arm = np.logical_and(visible_arm, valid_arm)


    occluded_hand_jnt = np.array(data['occluded_hand_jnt'], dtype=bool)
    visible_hand_jnt = np.logical_not(occluded_hand_jnt)

    invisible_arm = np.logical_not(visible_arm)
    invisible_hand = np.logical_not(visible_hand)

    HAND_ACC_ERROR = compute_acceleration_error(gt_hand_j3d / 1000, pred_hand_j3d / 1000, visible_hand, fps=30.0)
    ARM_ACC_ERROR = compute_acceleration_error(gt_arm_j3d / 1000, pred_arm_j3d / 1000, visible_arm, fps=30.0)

    HAND_IN_ACC_ERROR = compute_acceleration_error(gt_hand_j3d / 1000, pred_hand_j3d / 1000, invisible_hand, fps=30.0)
    ARM_IN_ACC_ERROR = compute_acceleration_error(gt_arm_j3d / 1000, pred_arm_j3d / 1000, invisible_arm, fps=30.0)

    HAND_CS_MPJPE, HAND_RR_MPJPE, HAND_PA_MPJPE = compute_3d_errors_batch(gt_hand_j3d, pred_hand_j3d, visible_hand, root_joint=0)
    HAND_IN_CS_MPJPE, HAND_IN_RR_MPJPE, HAND_IN_PA_MPJPE = compute_3d_errors_batch(gt_hand_j3d, pred_hand_j3d, invisible_hand, root_joint=0)

    HAND_OCC_CS_MPJPE, HAND_OCC_RR_MPJPE, HAND_OCC_PA_MPJPE, HAND_OCC_COUNT = compute_occ_3d_errors_batch(gt_hand_j3d, pred_hand_j3d, visible_hand, occluded_hand_jnt, root_joint=0)
    HAND_VIS_CS_MPJPE, HAND_VIS_RR_MPJPE, HAND_VIS_PA_MPJPE, HAND_VIS_COUNT = compute_occ_3d_errors_batch(gt_hand_j3d, pred_hand_j3d, visible_hand, visible_hand_jnt, root_joint=0)

    ARM_CS_MPJPE, ARM_RR_MPJPE, ARM_PA_MPJPE = compute_3d_errors_batch(gt_arm_j3d, pred_arm_j3d, visible_arm, root_joint=2)
    ARM_IN_CS_MPJPE, ARM_IN_RR_MPJPE, ARM_IN_PA_MPJPE = compute_3d_errors_batch(gt_arm_j3d, pred_arm_j3d, invisible_arm, root_joint=2)

    # --- NEW: Hand metrics conditioned on arm visibility, GIVEN hand is visible ---
    mask_hand_vis__arm_vis = np.logical_and(visible_hand, visible_arm)
    mask_hand_vis__arm_inv = np.logical_and(visible_hand, np.logical_not(visible_arm))

    # Acceleration error (hand) under the two conditions (convert to meters inside the function call)
    HAND_ACC_WHEN_ARM_VIS = compute_acceleration_error(gt_hand_j3d / 1000, pred_hand_j3d / 1000, mask_hand_vis__arm_vis, fps=30.0)
    HAND_ACC_WHEN_ARM_INV = compute_acceleration_error(gt_hand_j3d / 1000, pred_hand_j3d / 1000, mask_hand_vis__arm_inv, fps=30.0)

    # 3D errors (hand) under the two conditions
    HV_AV_CS, HV_AV_RR, HV_AV_PA = compute_3d_errors_batch(gt_hand_j3d, pred_hand_j3d, mask_hand_vis__arm_vis, root_joint=0)
    HV_AI_CS, HV_AI_RR, HV_AI_PA = compute_3d_errors_batch(gt_hand_j3d, pred_hand_j3d, mask_hand_vis__arm_inv, root_joint=0)

    HAND_CS_MPJPE, HAND_ACC_ERROR, FAILURE_RATE = _filter_hand_failure_cases(
        HAND_CS_MPJPE,
        HAND_ACC_ERROR,
        ignore_failure_solves,
    )
        
    HAND_CS_MPJPE = np.mean(HAND_CS_MPJPE)
    HAND_RR_MPJPE = np.mean(HAND_RR_MPJPE) 
    HAND_PA_MPJPE = np.mean(HAND_PA_MPJPE)

    HAND_IN_CS_MPJPE = np.mean(HAND_IN_CS_MPJPE)
    HAND_IN_RR_MPJPE = np.mean(HAND_IN_RR_MPJPE)
    HAND_IN_PA_MPJPE = np.mean(HAND_IN_PA_MPJPE)
    
    HAND_OCC_CS_MPJPE = np.mean(HAND_OCC_CS_MPJPE)
    HAND_OCC_RR_MPJPE = np.mean(HAND_OCC_RR_MPJPE)
    HAND_OCC_PA_MPJPE = np.mean(HAND_OCC_PA_MPJPE)

    HAND_VIS_CS_MPJPE = np.mean(HAND_VIS_CS_MPJPE)
    HAND_VIS_RR_MPJPE = np.mean(HAND_VIS_RR_MPJPE)
    HAND_VIS_PA_MPJPE = np.mean(HAND_VIS_PA_MPJPE)

    ARM_CS_MPJPE = np.mean(ARM_CS_MPJPE)
    ARM_RR_MPJPE = np.mean(ARM_RR_MPJPE) 
    ARM_PA_MPJPE = np.mean(ARM_PA_MPJPE)

    HAND_ACC_ERROR = np.mean(HAND_ACC_ERROR)
    ARM_ACC_ERROR = np.mean(ARM_ACC_ERROR)

    ARM_IN_CS_MPJPE = np.mean(ARM_IN_CS_MPJPE)
    ARM_IN_RR_MPJPE = np.mean(ARM_IN_RR_MPJPE)
    ARM_IN_PA_MPJPE = np.mean(ARM_IN_PA_MPJPE)

    HAND_IN_ACC_ERROR = np.mean(HAND_IN_ACC_ERROR)
    ARM_IN_ACC_ERROR = np.mean(ARM_IN_ACC_ERROR)

    CS_WHEN_ARM_VIS_GIVEN_HAND_VIS = np.mean(HV_AV_CS)
    RR_WHEN_ARM_VIS_GIVEN_HAND_VIS = np.mean(HV_AV_RR)
    PA_WHEN_ARM_VIS_GIVEN_HAND_VIS = np.mean(HV_AV_PA)
    ACC_WHEN_ARM_VIS_GIVEN_HAND_VIS = np.mean(HAND_ACC_WHEN_ARM_VIS)

    CS_WHEN_ARM_INV_GIVEN_HAND_VIS = np.mean(HV_AI_CS)
    RR_WHEN_ARM_INV_GIVEN_HAND_VIS = np.mean(HV_AI_RR)
    PA_WHEN_ARM_INV_GIVEN_HAND_VIS = np.mean(HV_AI_PA)
    ACC_WHEN_ARM_INV_GIVEN_HAND_VIS = np.mean(HAND_ACC_WHEN_ARM_INV)

    return {
        'hand': {
            'ACC_ERROR': HAND_ACC_ERROR,
            'INVISIBLE_ACC_ERROR': HAND_IN_ACC_ERROR,

            'CS_MPJPE': HAND_CS_MPJPE,
            'RR_MPJPE': HAND_RR_MPJPE,
            'PA_MPJPE': HAND_PA_MPJPE,

            'INVISIBLE_CS_MPJPE': HAND_IN_CS_MPJPE,
            'INVISIBLE_RR_MPJPE': HAND_IN_RR_MPJPE,
            'INVISIBLE_PA_MPJPE': HAND_IN_PA_MPJPE,

            'OCC_CS_MPJPE': HAND_OCC_CS_MPJPE,
            'OCC_RR_MPJPE': HAND_OCC_RR_MPJPE,
            'OCC_PA_MPJPE': HAND_OCC_PA_MPJPE,

            'VIS_CS_MPJPE': HAND_VIS_CS_MPJPE,
            'VIS_RR_MPJPE': HAND_VIS_RR_MPJPE,
            'VIS_PA_MPJPE': HAND_VIS_PA_MPJPE,

            'OCC_CNT': HAND_OCC_COUNT,
            'VIS_CNT': HAND_VIS_COUNT,

            'FAILURE_RATE': FAILURE_RATE,
            
            # --- NEW keys (hand metrics conditioned on arm visibility, given hand is visible) ---
            'CS_MPJPE_WHEN_ARM_VISIBLE_GIVEN_HAND_VISIBLE': CS_WHEN_ARM_VIS_GIVEN_HAND_VIS,
            'RR_MPJPE_WHEN_ARM_VISIBLE_GIVEN_HAND_VISIBLE': RR_WHEN_ARM_VIS_GIVEN_HAND_VIS,
            'PA_MPJPE_WHEN_ARM_VISIBLE_GIVEN_HAND_VISIBLE': PA_WHEN_ARM_VIS_GIVEN_HAND_VIS,
            'ACC_ERROR_WHEN_ARM_VISIBLE_GIVEN_HAND_VISIBLE': ACC_WHEN_ARM_VIS_GIVEN_HAND_VIS,

            'CS_MPJPE_WHEN_ARM_INVISIBLE_GIVEN_HAND_VISIBLE': CS_WHEN_ARM_INV_GIVEN_HAND_VIS,
            'RR_MPJPE_WHEN_ARM_INVISIBLE_GIVEN_HAND_VISIBLE': RR_WHEN_ARM_INV_GIVEN_HAND_VIS,
            'PA_MPJPE_WHEN_ARM_INVISIBLE_GIVEN_HAND_VISIBLE': PA_WHEN_ARM_INV_GIVEN_HAND_VIS,
            'ACC_ERROR_WHEN_ARM_INVISIBLE_GIVEN_HAND_VISIBLE': ACC_WHEN_ARM_INV_GIVEN_HAND_VIS,

        },
        'arm': {
            'ACC_ERROR': ARM_ACC_ERROR,
            'INVISIBLE_ACC_ERROR': ARM_IN_ACC_ERROR,

            'CS_MPJPE': ARM_CS_MPJPE,
            'RR_MPJPE': ARM_RR_MPJPE,
            'PA_MPJPE': ARM_PA_MPJPE,

            'INVISIBLE_CS_MPJPE': ARM_IN_CS_MPJPE,
            'INVISIBLE_RR_MPJPE': ARM_IN_RR_MPJPE,
            'INVISIBLE_PA_MPJPE': ARM_IN_PA_MPJPE,
        },

    }


def evaluate_batch_two_hand(left_data, right_data):
    left_pred_hand_j3d = left_data['pred_hand_j3d'] * 1000 # to mm
    right_pred_hand_j3d = right_data['pred_hand_j3d'] * 1000 # to mm
   
    left_gt_hand_j3d = left_data['gt_hand_j3d'] * 1000 # to mm
    right_gt_hand_j3d = right_data['gt_hand_j3d'] * 1000 # to mm
   
    left_visible_hand = np.array(left_data['visible_hand'], dtype=bool)
    left_valid_hand = np.array(left_data['valid_hand_j3d'], dtype=bool)
    left_visible_hand = np.logical_and(left_visible_hand, left_valid_hand)

    right_visible_hand = np.array(right_data['visible_hand'], dtype=bool)
    right_valid_hand = np.array(right_data['valid_hand_j3d'], dtype=bool)
    right_visible_hand = np.logical_and(right_visible_hand, right_valid_hand)

    gt_hand_j3d = left_gt_hand_j3d[:, :1, :] - right_gt_hand_j3d[:, :1, :]
    pred_hand_j3d = left_pred_hand_j3d[:, :1, :] - right_pred_hand_j3d[:, :1, :]

    
    visible_hand = np.logical_and(left_visible_hand, right_visible_hand)
    invisible_hand = np.logical_not(visible_hand)

    HAND_ACC_ERROR = compute_acceleration_error(gt_hand_j3d / 1000, pred_hand_j3d / 1000, visible_hand, fps=30.0)
    HAND_IN_ACC_ERROR = compute_acceleration_error(gt_hand_j3d / 1000, pred_hand_j3d / 1000, invisible_hand, fps=30.0)

    HAND_CS_MPJPE, HAND_RR_MPJPE, HAND_PA_MPJPE = compute_3d_errors_batch(gt_hand_j3d, pred_hand_j3d, visible_hand, root_joint=0)
    HAND_IN_CS_MPJPE, HAND_IN_RR_MPJPE, HAND_IN_PA_MPJPE = compute_3d_errors_batch(gt_hand_j3d, pred_hand_j3d, invisible_hand, root_joint=0)

    HAND_CS_MPJPE = np.mean(HAND_CS_MPJPE)
    HAND_RR_MPJPE = np.mean(HAND_RR_MPJPE) 
    HAND_PA_MPJPE = np.mean(HAND_PA_MPJPE)

    HAND_IN_CS_MPJPE = np.mean(HAND_IN_CS_MPJPE)
    HAND_IN_RR_MPJPE = np.mean(HAND_IN_RR_MPJPE)
    HAND_IN_PA_MPJPE = np.mean(HAND_IN_PA_MPJPE)
    
    HAND_ACC_ERROR = np.mean(HAND_ACC_ERROR)

    HAND_IN_ACC_ERROR = np.mean(HAND_IN_ACC_ERROR)


    return {
        'hand': {
            'ACC_ERROR': HAND_ACC_ERROR,
            'INVISIBLE_ACC_ERROR': HAND_IN_ACC_ERROR,

            'CS_MPJPE': HAND_CS_MPJPE,
            'RR_MPJPE': HAND_RR_MPJPE,
            'PA_MPJPE': HAND_PA_MPJPE,

            'INVISIBLE_CS_MPJPE': HAND_IN_CS_MPJPE,
            'INVISIBLE_RR_MPJPE': HAND_IN_RR_MPJPE,
            'INVISIBLE_PA_MPJPE': HAND_IN_PA_MPJPE,
        },
    }




def evaluate_batch_hand(data, ignore_failure_solves=False):
    pred_hand_j3d = data['pred_hand_j3d'] * 1000 # to mm
    pred_arm_j3d = data['pred_arm_j3d'] * 1000 # to mm

    gt_hand_j3d = data['gt_hand_j3d'] * 1000 # to mm
    gt_arm_j3d = data['gt_arm_j3d'] * 1000 # to mm

    visible_hand = np.array(data['visible_hand'], dtype=bool)
    valid_hand = np.array(data['valid_hand_j3d'], dtype=bool)

    visible_hand = np.logical_and(visible_hand, valid_hand)


    occluded_hand_jnt = np.array(data['occluded_hand_jnt'], dtype=bool)
    visible_hand_jnt = np.logical_not(occluded_hand_jnt)

    invisible_hand = np.logical_not(visible_hand)

    HAND_ACC_ERROR = compute_acceleration_error(gt_hand_j3d / 1000, pred_hand_j3d / 1000, visible_hand, fps=30.0)

    HAND_IN_ACC_ERROR = compute_acceleration_error(gt_hand_j3d / 1000, pred_hand_j3d / 1000, invisible_hand, fps=30.0)

    HAND_CS_MPJPE, HAND_RR_MPJPE, HAND_PA_MPJPE = compute_3d_errors_batch(gt_hand_j3d, pred_hand_j3d, visible_hand, root_joint=0)
    HAND_IN_CS_MPJPE, HAND_IN_RR_MPJPE, HAND_IN_PA_MPJPE = compute_3d_errors_batch(gt_hand_j3d, pred_hand_j3d, invisible_hand, root_joint=0)

    HAND_OCC_CS_MPJPE, HAND_OCC_RR_MPJPE, HAND_OCC_PA_MPJPE, HAND_OCC_COUNT = compute_occ_3d_errors_batch(gt_hand_j3d, pred_hand_j3d, visible_hand, occluded_hand_jnt, root_joint=0)
    HAND_VIS_CS_MPJPE, HAND_VIS_RR_MPJPE, HAND_VIS_PA_MPJPE, HAND_VIS_COUNT = compute_occ_3d_errors_batch(gt_hand_j3d, pred_hand_j3d, visible_hand, visible_hand_jnt, root_joint=0)

    HAND_CS_MPJPE, HAND_ACC_ERROR, FAILURE_RATE = _filter_hand_failure_cases(
        HAND_CS_MPJPE,
        HAND_ACC_ERROR,
        ignore_failure_solves,
    )

    HAND_CS_MPJPE = np.mean(HAND_CS_MPJPE)
    HAND_RR_MPJPE = np.mean(HAND_RR_MPJPE) 
    HAND_PA_MPJPE = np.mean(HAND_PA_MPJPE)

    HAND_IN_CS_MPJPE = np.mean(HAND_IN_CS_MPJPE)
    HAND_IN_RR_MPJPE = np.mean(HAND_IN_RR_MPJPE)
    HAND_IN_PA_MPJPE = np.mean(HAND_IN_PA_MPJPE)
    
    HAND_OCC_CS_MPJPE = np.mean(HAND_OCC_CS_MPJPE)
    HAND_OCC_RR_MPJPE = np.mean(HAND_OCC_RR_MPJPE)
    HAND_OCC_PA_MPJPE = np.mean(HAND_OCC_PA_MPJPE)

    HAND_VIS_CS_MPJPE = np.mean(HAND_VIS_CS_MPJPE)
    HAND_VIS_RR_MPJPE = np.mean(HAND_VIS_RR_MPJPE)
    HAND_VIS_PA_MPJPE = np.mean(HAND_VIS_PA_MPJPE)

    HAND_ACC_ERROR = np.mean(HAND_ACC_ERROR)

    HAND_IN_ACC_ERROR = np.mean(HAND_IN_ACC_ERROR)


    return {
        'hand': {
            'ACC_ERROR': HAND_ACC_ERROR,
            'INVISIBLE_ACC_ERROR': HAND_IN_ACC_ERROR,

            'CS_MPJPE': HAND_CS_MPJPE,
            'RR_MPJPE': HAND_RR_MPJPE,
            'PA_MPJPE': HAND_PA_MPJPE,

            'INVISIBLE_CS_MPJPE': HAND_IN_CS_MPJPE,
            'INVISIBLE_RR_MPJPE': HAND_IN_RR_MPJPE,
            'INVISIBLE_PA_MPJPE': HAND_IN_PA_MPJPE,

            'OCC_CS_MPJPE': HAND_OCC_CS_MPJPE,
            'OCC_RR_MPJPE': HAND_OCC_RR_MPJPE,
            'OCC_PA_MPJPE': HAND_OCC_PA_MPJPE,

            'VIS_CS_MPJPE': HAND_VIS_CS_MPJPE,
            'VIS_RR_MPJPE': HAND_VIS_RR_MPJPE,
            'VIS_PA_MPJPE': HAND_VIS_PA_MPJPE,

            'OCC_CNT': HAND_OCC_COUNT,
            'VIS_CNT': HAND_VIS_COUNT,

            'FAILURE_RATE': FAILURE_RATE,
        },
    }


def compute_batch_accelaration_error(data):
    pred_hand_j3d = data['pred_hand_j3d'] * 1000 # to mm
    pred_arm_j3d = data['pred_arm_j3d'] * 1000 # to mm

    gt_hand_j3d = data['gt_hand_j3d'] * 1000 # to mm
    gt_arm_j3d = data['gt_arm_j3d'] * 1000 # to mm

    visible_hand = np.array(data['visible_hand'], dtype=bool)
    valid_hand = np.array(data['valid_hand_j3d'], dtype=bool)

    visible_hand = np.logical_and(visible_hand, valid_hand)


    occluded_hand_jnt = np.array(data['occluded_hand_jnt'], dtype=bool)
    visible_hand_jnt = np.logical_not(occluded_hand_jnt)

    invisible_hand = np.logical_not(visible_hand)

    HAND_ACC_ERROR = compute_acceleration_error(gt_hand_j3d / 1000, pred_hand_j3d / 1000, visible_hand, fps=30.0)
    HAND_IN_ACC_ERROR = compute_acceleration_error(gt_hand_j3d / 1000, pred_hand_j3d / 1000, invisible_hand, fps=30.0)

    print(HAND_ACC_ERROR.shape)
    exit(0)

    HAND_CS_MPJPE, HAND_RR_MPJPE, HAND_PA_MPJPE = compute_3d_errors_batch(gt_hand_j3d, pred_hand_j3d, visible_hand, root_joint=0)
    HAND_IN_CS_MPJPE, HAND_IN_RR_MPJPE, HAND_IN_PA_MPJPE = compute_3d_errors_batch(gt_hand_j3d, pred_hand_j3d, invisible_hand, root_joint=0)

    HAND_OCC_CS_MPJPE, HAND_OCC_RR_MPJPE, HAND_OCC_PA_MPJPE, HAND_OCC_COUNT = compute_occ_3d_errors_batch(gt_hand_j3d, pred_hand_j3d, visible_hand, occluded_hand_jnt, root_joint=0)
    HAND_VIS_CS_MPJPE, HAND_VIS_RR_MPJPE, HAND_VIS_PA_MPJPE, HAND_VIS_COUNT = compute_occ_3d_errors_batch(gt_hand_j3d, pred_hand_j3d, visible_hand, visible_hand_jnt, root_joint=0)

    HAND_CS_MPJPE = np.mean(HAND_CS_MPJPE)
    HAND_RR_MPJPE = np.mean(HAND_RR_MPJPE) 
    HAND_PA_MPJPE = np.mean(HAND_PA_MPJPE)

    HAND_IN_CS_MPJPE = np.mean(HAND_IN_CS_MPJPE)
    HAND_IN_RR_MPJPE = np.mean(HAND_IN_RR_MPJPE)
    HAND_IN_PA_MPJPE = np.mean(HAND_IN_PA_MPJPE)
    
    HAND_OCC_CS_MPJPE = np.mean(HAND_OCC_CS_MPJPE)
    HAND_OCC_RR_MPJPE = np.mean(HAND_OCC_RR_MPJPE)
    HAND_OCC_PA_MPJPE = np.mean(HAND_OCC_PA_MPJPE)

    HAND_VIS_CS_MPJPE = np.mean(HAND_VIS_CS_MPJPE)
    HAND_VIS_RR_MPJPE = np.mean(HAND_VIS_RR_MPJPE)
    HAND_VIS_PA_MPJPE = np.mean(HAND_VIS_PA_MPJPE)

    HAND_ACC_ERROR = np.mean(HAND_ACC_ERROR)

    HAND_IN_ACC_ERROR = np.mean(HAND_IN_ACC_ERROR)


    return {
        'hand': {
            'ACC_ERROR': HAND_ACC_ERROR,
            'INVISIBLE_ACC_ERROR': HAND_IN_ACC_ERROR,

            'CS_MPJPE': HAND_CS_MPJPE,
            'RR_MPJPE': HAND_RR_MPJPE,
            'PA_MPJPE': HAND_PA_MPJPE,

            'INVISIBLE_CS_MPJPE': HAND_IN_CS_MPJPE,
            'INVISIBLE_RR_MPJPE': HAND_IN_RR_MPJPE,
            'INVISIBLE_PA_MPJPE': HAND_IN_PA_MPJPE,

            'OCC_CS_MPJPE': HAND_OCC_CS_MPJPE,
            'OCC_RR_MPJPE': HAND_OCC_RR_MPJPE,
            'OCC_PA_MPJPE': HAND_OCC_PA_MPJPE,

            'VIS_CS_MPJPE': HAND_VIS_CS_MPJPE,
            'VIS_RR_MPJPE': HAND_VIS_RR_MPJPE,
            'VIS_PA_MPJPE': HAND_VIS_PA_MPJPE,

            'OCC_CNT': HAND_OCC_COUNT,
            'VIS_CNT': HAND_VIS_COUNT,
        },
    }
