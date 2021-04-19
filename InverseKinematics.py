import torch.optim as optim
from Loss import *
from compute_models import *
from get_subjects import *
from pyTorch_3Dviewer import *
torch.manual_seed(1510)


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


def get_goal_sequences(goal_joints, samples, indices=np.arange(59)):
    sequences = []
    for sample in samples:
        goals = []
        for pose in sample:
            pose = remove_excluded(pose, indices, datatype='dict', type='numpy')
            goals.append(set_goal(goal_joints, pose))
        sequences.append(goals)
    return sequences


def unpack_sequence(joint, sequence):
    coordinate_list = []
    for goal in sequence:
        coordinate_list.append(goal[joint])
    return coordinate_list

class Inverse_model:
    def __init__(self, prior = (None, None), indices=np.arange(59), saveframes=True, plot=False, pose=None):
        self.prior = prior
        if self.prior[0] == 'hmmGauss':
            self.update = True
        else:
            self.update = False
        self.indices = indices
        self.saveframes = saveframes
        self.frames = []
        self.plot = plot
        self.joints, dummy_pose, dof = dummy(return_dof=True)
        if not pose:
            self.pose = dummy_pose
        else:
            self.pose = pose
        self.joints['root'].set_motion(self.pose)


    def stitch(self, optim_joints):
        pose = []
        idx = 0
        for i in range(59):
            if i in self.indices:
                pose.append(optim_joints[idx])
                idx += 1
            else:
                pose.append(torch.tensor([0.], requires_grad=False))
        return torch.cat(pose)


    def inverse_kinematics(self, goal, lr=1, n_epochs=100, lh_var=1e-2, weight_decay=0):
        #saveframes = self.saveframes
        frames = []

        Loss_function = get_loss_func(self.prior, lh_var)

        if self.update:
            Loss_function.update_weights(pose2array(self.pose)[self.indices].reshape(1, -1))

        # initialize as tensor of zeros
        optim_joints = [torch.tensor([pose2array(self.pose)[i]], requires_grad=True) for i in self.indices]

        optimizer = optim.Adam(optim_joints, lr=lr, weight_decay=weight_decay)

        y = [y for y in goal.values()]

        for epoch in range(n_epochs+1):
            def closure():
                optimizer.zero_grad()

                pose = self.stitch(optim_joints)
                pose_dict = array2pose(pose)

                self.joints['root'].set_motion(pose_dict)
                yhat = [self.joints[goal_joint].coordinate for goal_joint in goal.keys()]

                if self.saveframes:
                    frames.append(pose_dict)

                loss = Loss_function.Loss(yhat, y, pose[self.indices])
                loss.backward(retain_graph=True)
                return loss
            optimizer.step(closure)
        self.pose = array2pose(self.stitch(optim_joints).detach())
        self.frames = frames
        if self.plot:
            draw_cluster(torch.cat(optim_joints), self.indices)