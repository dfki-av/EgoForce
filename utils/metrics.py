import numpy as np
import traceback


def compute_similarity_transform(S1, S2):
    '''
    Computes a similarity transform (sR, t) that takes
    a set of 3D points S1 (3 x N) closest to a set of 3D points S2,
    where R is an 3x3 rotation matrix, t 3x1 translation, s scale.
    i.e. solves the orthogonal Procrutes problem.
    '''
    transposed = False
    if S1.shape[0] != 3 and S1.shape[0] != 2:
        S1 = S1.T
        S2 = S2.T
        transposed = True
    assert(S2.shape[1] == S1.shape[1])

    # 1. Remove mean.
    mu1 = S1.mean(axis=1, keepdims=True)
    mu2 = S2.mean(axis=1, keepdims=True)
    X1 = S1 - mu1
    X2 = S2 - mu2

    # 2. Compute variance of X1 used for scale.
    var1 = np.sum(X1**2)

    # 3. The outer product of X1 and X2.
    K = X1.dot(X2.T)

    # 4. Solution that Maximizes trace(R'K) is R=U*V', where U, V are
    # singular vectors of K.
    U, s, Vh = np.linalg.svd(K)
    V = Vh.T
    # Construct Z that fixes the orientation of R to get det(R)=1.
    Z = np.eye(U.shape[0])
    Z[-1, -1] *= np.sign(np.linalg.det(U.dot(V.T)))
    # Construct R.
    R = V.dot(Z.dot(U.T))

    # 5. Recover scale.
    scale = np.trace(R.dot(K)) / var1

    # 6. Recover translation.
    t = mu2 - scale*(R.dot(mu1))

    # 7. Error:
    S1_hat = scale*R.dot(S1) + t

    if transposed:
        S1_hat = S1_hat.T

    return S1_hat


def compute_similarity_transform_batch(S1_batch, S2_batch):
    batch_size = S1_batch.shape[0]
    assert S1_batch.shape == S2_batch.shape

    transformed_batch = np.empty_like(S1_batch)

    count = 0
    for i in range(batch_size):
        S1 = S1_batch[i]
        S2 = S2_batch[i]

        try:
            transformed_batch[i] = compute_similarity_transform(S1, S2)
            count += 1
        except Exception as e:
            transformed_batch[i] = S2  

            traceback.print_exc()
            print(f"Error processing batch {i}: {e}")

    return transformed_batch, count


def align_by_pelvis(joints, return_pelvis=False):
    left_id = 12
    right_id = 8

    pelvis = (joints[left_id, :] + joints[right_id, :]) / 2.0
    pelvis = pelvis[None, ...]
    
    rr_joints = joints - pelvis
    
    if return_pelvis:
        return rr_joints, pelvis

    return rr_joints


def align_by_pelvis_batch(joints, return_pelvis=False):
    left_id = 12
    right_id = 8

    pelvis = (joints[:, left_id, :] + joints[:, right_id, :]) / 2.0 # middle of left and right hip
    pelvis = pelvis[:, None, ...]
    
    rr_joints = joints - pelvis

    if return_pelvis:
        return rr_joints, pelvis

    return rr_joints


# def compute_errors(gt3ds, preds):
#     """
#     Gets MPJPE after pelvis alignment + MPJPE after Procrustes.
#     Evaluates on the 14 common joints.
#     Inputs:
#       - gt3ds: N x 14 x 3
#       - preds: N x 14 x 3
#     """
#     errors, errors_pa = [], []
#     for i, (gt3d, pred) in enumerate(zip(gt3ds, preds)):
#         gt3d = gt3d.reshape(-1, 3)
#         # Root align.
#         gt3d = align_by_pelvis(gt3d)
#         pred3d = align_by_pelvis(pred)

#         joint_error = np.sqrt(np.sum((gt3d - pred3d)**2, axis=1))
#         errors.append(np.mean(joint_error))

#         # Get PA error.
#         pred3d_sym = compute_similarity_transform(pred3d, gt3d)
#         pa_error = np.sqrt(np.sum((gt3d - pred3d_sym)**2, axis=1))
#         errors_pa.append(np.mean(pa_error))

#     return errors, errors_pa



def compute_3d_errors_joints(gt3ds, preds, valid_j3d):
    valid_j3d = np.sum(valid_j3d, -1).mean(-1)
    valid_j3d = valid_j3d > 0
    
    gt3ds = gt3ds[valid_j3d, :, :]
    preds = preds[valid_j3d, :, :]
    
    cnt = gt3ds.shape[1]
    joint_error = np.sqrt(np.sum((gt3ds - preds)**2, axis=-1))
    errors = np.sum(joint_error, 1) / cnt


    pred3d_sym, _ = compute_similarity_transform_batch(preds, gt3ds)
    joint_error = np.sqrt(np.sum((gt3ds - pred3d_sym)**2, axis=-1))
    errors_pa = np.sum(joint_error, 1) / cnt

    return errors, errors_pa


def compute_3d_errors_batch(gt3ds, preds, valid_j3d, root_joint):
    valid_j3d = valid_j3d > 0
    
    if not np.count_nonzero(valid_j3d):
        return np.zeros(gt3ds.shape[1]), np.zeros(gt3ds.shape[1]), np.zeros(gt3ds.shape[1])

    gt3ds = gt3ds[valid_j3d, :, :]
    preds = preds[valid_j3d, :, :]
    
    cnt = gt3ds.shape[0]
    
    joint_error = np.sqrt(np.sum((gt3ds - preds)**2, axis=-1))
    errors = np.sum(joint_error, 0) / cnt
    
    joint_error_rr = np.sqrt(np.sum(((gt3ds - gt3ds[:, root_joint:root_joint+1]) - (preds - preds[:, root_joint:root_joint+1]))**2, axis=-1))
    errors_rr = np.sum(joint_error_rr, 0) / cnt
    
    pred3d_sym, _ = compute_similarity_transform_batch(preds, gt3ds)
    joint_error = np.sqrt(np.sum((gt3ds - pred3d_sym)**2, axis=-1))
    errors_pa = np.sum(joint_error, 0) / cnt

    return errors, errors_rr, errors_pa


def compute_occ_3d_errors_batch(gt3ds, preds, valid_j3d, occluded_label, root_joint):
    valid_j3d = valid_j3d > 0
    
    occluded_label = occluded_label[valid_j3d, :]
    gt3ds = gt3ds[valid_j3d, :, :]
    preds = preds[valid_j3d, :, :]

    occluded_cnt = occluded_label.sum()

    if not np.count_nonzero(valid_j3d) or not occluded_cnt:
        return np.zeros(gt3ds.shape[1]), np.zeros(gt3ds.shape[1]), np.zeros(gt3ds.shape[1]), 0

    joint_error = np.sqrt(np.sum((gt3ds - preds)**2, axis=-1))   
    errors = (joint_error * occluded_label).sum() / occluded_cnt
     
    joint_error_rr = np.sqrt(np.sum(((gt3ds - gt3ds[:, root_joint:root_joint+1]) - (preds - preds[:, root_joint:root_joint+1]))**2, axis=-1))
    errors_rr = (joint_error_rr * occluded_label).sum() / occluded_cnt

    pred3d_sym, cnt = compute_similarity_transform_batch(preds, gt3ds)
    joint_error = np.sqrt(np.sum((gt3ds - pred3d_sym)**2, axis=-1))
    errors_pa = (joint_error * occluded_label).sum() / occluded_cnt

    return errors, errors_rr, errors_pa, occluded_cnt


def compute_acceleration_error(gt3ds, preds, valid_j3d, fps=30.0):
    valid_j3d = valid_j3d > 0

    # Compute acceleration (second-order difference)
    gt_acc = gt3ds[2:] - 2 * gt3ds[1:-1] + gt3ds[:-2]
    pred_acc = preds[2:] - 2 * preds[1:-1] + preds[:-2]

    # Mask invalid joints
    valid_mask = valid_j3d[2:] & valid_j3d[1:-1] & valid_j3d[:-2]
    h = 1 / fps  # stencil width

    gt_acc = gt_acc[valid_mask, :, :] / (h**2)
    pred_acc = pred_acc[valid_mask, :, :] / (h**2)

    # Compute Euclidean distance
    acc_error = np.sqrt(np.sum((gt_acc - pred_acc) ** 2, axis=-1))

    valid_cnt = valid_mask.sum()
    if valid_cnt == 0: return np.zeros(gt_acc.shape[1])

    acc_error = np.sum(acc_error, 0) / valid_cnt
    
    return acc_error


