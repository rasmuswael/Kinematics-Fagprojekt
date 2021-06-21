import torch.optim as optim
from inverse_kinematics.Loss import *
from inverse_kinematics.compute_models import *
from inverse_kinematics.better_3Dviewer import *
torch.manual_seed(1510)


def set_goal(goal_joints, pose):
    '''
    Sets constraints (goal coordinates) given a pose.
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


def get_goal_sequences(goal_joints, samples, indices=np.arange(59), return_trunc_samples=False):
    """Helper function to get sequences of goal coordinates
    :return goal_sequences, sequences of goal coordinates
    :arg goal_joints, the goal joint to be tracked
    :arg samples, samples from which the goal points are extracted
    :arg indices, indices for the included joints
    :arg return_trunc_samples, whether to return truncated samples or not."""
    sequences = []
    trunc_samples_list = []
    for sample in samples:
        goals = []
        trunc_samples = []
        for pose in sample:
            posearr = pose2array(pose, type='numpy')
            posearr = truncate(np.vstack((posearr,posearr)))[0]
            pose = array2pose(remove_excluded(posearr, indices, type='numpy'), indices, type='numpy')
            goals.append(set_goal(goal_joints, pose))
            trunc_samples.append(pose)
        trunc_samples_list.append(trunc_samples)
        sequences.append(goals)
    if return_trunc_samples:
        return sequences, trunc_samples_list
    return sequences


def unpack_sequence(joint, sequence):
    coordinate_list = []
    for goal in sequence:
        coordinate_list.append(goal[joint].detach().numpy())
    return coordinate_list


class Inverse_model:
    def __init__(self, prior = (None, None), indices=np.arange(59), saveframes=True, plot=False):
        self.prior = prior
        if self.prior[0] == 'hmmGauss':
            self.update = True
        else:
            self.update = False
        self.indices = indices
        self.saveframes = saveframes
        self.frames = []
        self.plot = plot
        self.joints, self.dummy_pose, dof = dummy(return_dof=True)
        self.pose = self.dummy_pose
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


    def inverse_kinematics(self, goal, lr=1, n_epochs=100, lh_var=1e-2, weight_decay=0, opt_pose='dummy', print_loss=True):
        frames = []

        Loss_function = get_loss_func(self.prior, lh_var)

        if self.update:
            Loss_function.update_weights(pose2array(self.pose)[self.indices].reshape(1, -1))
            self.hstates = Loss_function.prior_model.weights

        if opt_pose == 'self':
            ipose = self.pose
        elif opt_pose == 'dummy':
            ipose = self.dummy_pose
        optim_joints = [torch.tensor([pose2array(ipose)[i]], requires_grad=True) for i in self.indices]

        optimizer = optim.Adam(optim_joints, lr=lr, weight_decay=weight_decay)

        y = torch.cat([y for y in goal.values()], 1)

        for epoch in range(n_epochs+1):
            def closure():
                optimizer.zero_grad()

                pose = self.stitch(optim_joints)
                pose_dict = array2pose(pose)

                self.joints['root'].set_motion(pose_dict)
                yhat = torch.cat([self.joints[goal_joint].coordinate for goal_joint in goal.keys()], 1)

                if self.saveframes:
                    frames.append(array2pose(pose.detach().numpy(), type='numpy'))

                loss = Loss_function.Loss(yhat, y, pose[self.indices])
                loss.backward(retain_graph=True)
                if epoch % 10 == 0 and print_loss:
                    print('step: {}, loss: {}'.format(epoch, loss.detach()))
                return loss
            optimizer.step(closure)

        # self.pose = poses[np.argmin(losses)]
        self.pose = array2pose(self.stitch(optim_joints).detach())
        self.frames = frames
        if self.plot:
            draw_cluster(torch.cat(optim_joints), self.indices)