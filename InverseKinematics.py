from dummy import *
import torch.optim as optim
from Loss import *

torch.manual_seed(1510)

est_name = 'rclavicle'
goal_name = 'rfoot'



dummy_joints, dummy_pose = dummy()

goal_pose = dummy_pose.copy()
goal_pose['rfemur'] = torch.tensor([-60, 0, 0])


dummy_joints['root'].set_motion(goal_pose)
goal_coord = dummy_joints[goal_name].coordinate

def inverse_kinematics(goal_name, goal_coord):
    dummy_joints, dummy_pose = dummy()
    dummy_joints['root'].set_motion(dummy_pose)
    y = goal_coord

    # We exclude some joints for optimisation
    children = [ 'root', 'lowerback','lfemur', 'ltibia','lfoot','rfemur','rtibia','rfoot','upperback','thorax', 'lowerneck',
                 'lclavicle', 'rclavicle','upperneck', 'head','lhumerus','lradius','lwrist','lhand', 'rhumerus','rradius', 'rwrist','rhand']

    # initialize as tensor of zeros
    OptimJoints = [torch.zeros(1, requires_grad=True), torch.zeros(1, requires_grad=True), torch.zeros(1, requires_grad=True)]
    n_angles = [3]

    for child in children[1:]:
        count = 0
        for lim in range(3):
            if dummy_joints[child].limits[lim].any():
                OptimJoints.append(torch.zeros(1, requires_grad=True))

                count += 1
            # if (dummy_joints[children[i]].limits[lim][0] == 0) and (dummy_joints[children[i]].limits[lim][1] == 0):
            #     pass
            # else:
            #     OptimJoints.append(torch.zeros(1, requires_grad=True))
            #
            #     count += 1
        n_angles.append(count)

    means, covs, queries = get_prior_model()
    mean, cov = means[0], covs[0]

    # Draw prior
    mean_pose = dummy_pose.copy()
    print(torch.cat([torch.zeros(3), torch.tensor(mean[:3])]))
    mean_pose['root'] = torch.cat([torch.zeros(3), torch.tensor(mean[:3])]).float()

    count = 2
    for i, child in enumerate(children[1:]):
        # est_pose[children[i]] = OptimJoints[i]-
        # mean_pose[child] = torch.cat(OptimJoints[count: count + n_angles[i + 1]])
        mean_pose[child] = torch.tensor(mean[count: count + n_angles[i + 1]])
        count += n_angles[i + 1]
    dummy_joints['root'].set_motion(mean_pose)
    dummy_joints['root'].draw()

    #make sure we remove unnecessary joints from prior!!
    jump = 2
    delete_indices = []
    for joint_name in dummy_pose.keys():
        prev_jump = jump
        if joint_name != 'root':
            joint_name
            jump += dummy_pose[joint_name].size()[0]
            if joint_name not in children:
                print(joint_name)
                delete_indices.append(list(range(prev_jump, jump)))
    delete_indices = [item for sublist in delete_indices for item in sublist]
    delete_indices = np.array(delete_indices)

    mean = np.delete(mean, delete_indices)
    cov = np.delete(np.delete(cov, delete_indices, 0), delete_indices, 1)
    prior = normal_prior(mean, cov)


    lr = 1
    n_epochs = 100
    optimizer = optim.Adam(OptimJoints, lr=lr)

    for epoch in range(n_epochs+1):
        if not epoch % 25:
            print(epoch)
            dummy_joints['root'].draw(goal_coord.detach().numpy())

        def closure():
            optimizer.zero_grad()


            est_pose = dummy_pose.copy()
            #loop and set new values
            est_pose['root'] = torch.cat([torch.zeros(3, requires_grad=False), torch.cat(OptimJoints[:3])])

            count = 2
            for i, child in enumerate(children[1:]):
                # est_pose[children[i]] = OptimJoints[i]-
                est_pose[child] = torch.cat(OptimJoints[count : count+n_angles[i+1]])
                count += n_angles[i+1]
            dummy_joints['root'].set_motion(est_pose)


            yhat = dummy_joints[goal_name].coordinate
            # loss = torch.cdist(yhat.transpose(0, 1), y.transpose(0, 1), p=2.0)
            loss = Loss_V1(torch.cat(OptimJoints), yhat, y, prior, 10)
            print(loss)
            loss.backward(retain_graph=True)
            return loss
        optimizer.step(closure)

        with torch.no_grad():
            for child in children:
                count = 0
                for lim in range(3):
                    if dummy_joints[child].limits[lim].any():
                        OptimJoints[count].clamp_(dummy_joints[child].limits[lim][0], dummy_joints[child].limits[lim][1])
                        count += 1

    print(OptimJoints)
    print(children)
if __name__ == '__main__':
    inverse_kinematics(goal_name, goal_coord)
