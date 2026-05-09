def get_limb(config, limb_model, global_orient, betas, hand_pose, transl, hand_type, arm_shape, arm_R):
    B, T = global_orient.shape[:2]
    BxT = B * T
    
    global_orient = global_orient.reshape(BxT, *global_orient.shape[2:])
    betas = betas.reshape(BxT, *betas.shape[2:])
    hand_pose = hand_pose.reshape(BxT, *hand_pose.shape[2:])
    transl = transl.reshape(BxT, *transl.shape[2:])
    hand_type = hand_type.reshape(B * T)
    arm_shape = arm_shape.reshape(BxT, *arm_shape.shape[2:])
    arm_R = arm_R.reshape(BxT, *arm_R.shape[2:])

    limb_output = limb_model(betas, global_orient, hand_pose, transl, hand_type, 
                             arm_shape, arm_R)

    return limb_output
