from dummy import *
import torch.optim as optim
import torch
from Loss import *

torch.manual_seed(1510)

est_name = 'rclavicle'
goal_name = 'rfoot'



dummy_joints, dummy_pose = dummy()

goal_pose = dummy_pose.copy()
goal_pose['rfemur'] = torch.tensor([60, 0, 0])


dummy_joints['root'].set_motion(goal_pose)
goal_coord = dummy_joints[goal_name].coordinate

def inverse_kinematics(goal_name, goal_coord):
    dummy_joints, dummy_pose = dummy()
    dummy_joints['root'].set_motion(dummy_pose)
    y = goal_coord

    children = [ 'lowerback','lfemur', 'ltibia','lfoot','rfemur','rtibia','rfoot','upperback','thorax', 'lowerneck',
                 'lclavicle', 'rclavicle','upperneck', 'head','lhumerus','lradius','lwrist','lhand', 'rhumerus','rradius', 'rwrist','rhand']

    # initialize as tensor of zeros
    OptimJoints = []
    n_angles = []
    means, covs, queries = get_prior_model()

    for i in range(len(children)):
        count = 0
        for lim in range(3):
            if (dummy_joints[children[i]].limits[lim][0] == 0) and (dummy_joints[children[i]].limits[lim][1] == 0):
                pass
            else:
                OptimJoints.append(torch.zeros(1, requires_grad= True))

                count += 1
        n_angles.append(count)

    lr = 1e-1
    n_epochs = 100
    optimizer = optim.Adam(OptimJoints, lr=1)

    for epoch in range(n_epochs+1):


        if not epoch % 20:
            print(epoch)
            dummy_joints['root'].draw(goal_coord.detach().numpy())

        def closure():
            optimizer.zero_grad()


            est_pose = dummy_pose.copy()
            #loop and set new values

            count = 0
            for i in range(len(children)):
                # est_pose[children[i]] = OptimJoints[i]
                est_pose[children[i]] = torch.cat(OptimJoints[count : count+n_angles[i]])
                count += n_angles[i]
            dummy_joints['root'].set_motion(est_pose)


            yhat = dummy_joints[goal_name].coordinate
            # loss = torch.cdist(yhat.transpose(0, 1), y.transpose(0, 1), p=2.0)
            loss = Loss_V1(torch.cat(list(est_pose.values()))[3:], yhat, y, normal_prior(means[0],covs[0]))
            loss.backward(retain_graph=True)
            return loss
        optimizer.step(closure)

        with torch.no_grad():
            count = 0
            for i in range(len(children)):
                for lim in range(3):
                    if (dummy_joints[children[i]].limits[lim][0] == 0) and (dummy_joints[children[i]].limits[lim][1] == 0):
                        pass
                    else:
                        OptimJoints[count].clamp_(dummy_joints[children[i]].limits[lim][0], dummy_joints[children[i]].limits[lim][1])
                        count += 1
    print(OptimJoints)
    print(children)
if __name__ == '__main__':
    inverse_kinematics(goal_name, goal_coord)
