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

class Inverse_model:
    def __init__(self, prior = (None, None), excluded=None, saveframes = True, plot=False):
        self.prior_type, self.prior_model = prior
        self.excluded = excluded
        self.children, self.indices = exclude(excluded, return_indices=True)
        self.saveframes = saveframes
        self.frames = []
        self.plot = plot
        self.result = None

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

    def inverse_kinematics(self, goal, lr=1, n_epochs=100):
        #saveframes = self.saveframes
        frames = []
        #prior_type, prior_model = self.prior_type, self.prior_model

        joints, dummy_pose, dof = dummy(return_dof=True)
        joints['root'].set_motion(dummy_pose)

        # initialize as tensor of zeros
        optim_joints = [torch.tensor([0.], requires_grad=True) for i in range(np.size(self.indices))]

        optimizer = optim.Adam(optim_joints, lr=lr)

        y = [y for y in goal.values()]

        for epoch in range(n_epochs+1):
            def closure():
                optimizer.zero_grad()

                pose = self.stitch(optim_joints)
                pose_dict = array2pose(pose)

                joints['root'].set_motion(pose_dict)
                yhat = [joints[goal_joint].coordinate for goal_joint in goal.keys()]

                if self.saveframes:
                    frames.append(pose_dict)

                loss = Loss(yhat, y, self.prior_model, self.prior_type, pose[self.indices])
                loss.backward(retain_graph=True)
                return loss
            optimizer.step(closure)
        self.result = optim_joints
        self.frames = frames