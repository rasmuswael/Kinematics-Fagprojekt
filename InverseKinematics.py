import torch.optim as optim
from Loss import *
from compute_models import *
from get_subjects import *
from pyTorch_3Dviewer import *
torch.manual_seed(1510)


# example input:
# goal_joints = ['rfoot']
# pose = {'rfemur': [-60, 0, 0]}


def set_goal(goal_joints, pose):
    '''
    Sets constraints.
    Returns dictionary of goal joints and their desired cartesian coordinates

    Arguments:
        goal_joints: type list. The joints that we wish to set constraints for
        pose: type dictionary. Any pose that produces the desired constraints for the goal joints. Not all joint angles
            have to be specified.
    '''
    dummy_joints, dummy_pose = dummy()

    for joint, angles in pose.items():
        dummy_pose[joint] = torch.tensor(angles)

    dummy_joints['root'].set_motion(dummy_pose)

    goal = {}
    for goal_joint in goal_joints:
        goal[goal_joint] = dummy_joints[goal_joint].coordinate
    return goal


def inverse_kinematics(goal, prior = (None,), excluded=None, saveframes = False, plot=False):
    prior_type, prior_model = prior

    frames = []

    dummy_joints, dummy_pose, dof = dummy(return_dof=True)
    dummy_joints['root'].set_motion(dummy_pose)

    # We exclude some joints for optimisation
    children, indices = exclude(excluded, return_indices=True)

    # initialize as tensor of zeros
    optim_joints = [torch.tensor([0.], requires_grad=True) for i in range(np.size(indices))]

    lr = 1
    n_epochs = 100
    optimizer = optim.Adam(optim_joints, lr=lr)

    y = [y for y in goal.values()]

    for epoch in range(n_epochs+1):
        if plot and not epoch % 10:
            print(epoch)
            dummy_joints['root'].draw(list(goal.values())[0].numpy())

        def closure():
            optimizer.zero_grad()

            #model outpout
            # pose = [torch.zeros(3).float()]
            pose = []
            idx = 0
            for i in range(59):
                if i in indices:
                    pose.append(optim_joints[idx])
                    idx += 1
                else:
                    pose.append(torch.tensor([0.], requires_grad=False))

            pose = torch.cat(pose)
            pose_dict = array2pose(pose)
            # print(pose)
            dummy_joints['root'].set_motion(pose_dict)
            yhat = [dummy_joints[goal_joint].coordinate for goal_joint in goal.keys()]

            if saveframes:
                frames.append(pose_dict)

            if prior_type == None:
                loss = Loss_Euclidian(yhat, y)
            elif prior_type == 'normal':
                loss = Loss_Normal(pose[indices], yhat, y, prior_model, 1e-2)
            print(loss)
            loss.backward(retain_graph=True)
            return loss
        optimizer.step(closure)

        # with torch.no_grad():
        #     for child in children:
        #         limits = dummy_joints[child].limits
        #         for
            # for child in children:
            #     count = 0
            #     for lim in range(3):
            #         if dummy_joints[child].limits[lim].any():
            #             OptimJoints[count].clamp_(dummy_joints[child].limits[lim][0], dummy_joints[child].limits[lim][1])
            #             count += 1

    print(optim_joints)
    print(children)
    if saveframes:
        return frames
if __name__ == '__main__':
    A = 123
    # inverse_kinematics(goal_name, goal_coord)
