from dummy import *
import torch.optim as optim
import torch
torch.manual_seed(1510)

est_name = 'rclavicle'
goal_name = 'rfoot'



dummy_joints, dummy_pose = dummy()

goal_pose = dummy_pose.copy()
goal_pose['rfemur'] = torch.tensor([60, 0, 0])


dummy_joints['root'].set_motion(goal_pose)
goal_coord = dummy_joints[goal_name].coordinate

def getChildren(childname, goal_name, childrenlist):
    children = childrenlist
    if childname == goal_name:
        return children
    currentchild = dummy_joints[childname].name
    children.append(currentchild)
    for i in dummy_joints[currentchild].children:
            return getChildren(i.name, goal_name, children)

def inverse_kinematics(goal_name, goal_coord,est_name, full_body):
    dummy_joints, dummy_pose = dummy()
    dummy_joints['root'].set_motion(dummy_pose)
    y = goal_coord
    #L = torch.cdist(yhat.transpose(0,1), y.transpose(0,1), p=2.0)
    #yhat = dummy_joints[goal_name].coordinate

    #fill children list with children
    if full_body == False:
        children = getChildren(est_name, goal_name, childrenlist = list())

    else:
        children = [ 'lhipjoint' ,'rhipjoint', 'lowerback','lfemur', 'ltibia','lfoot','rfemur','rtibia','rfoot','upperback','thorax', 'lowerneck',
                     'lclavicle', 'rclavicle','upperneck', 'head','lhumerus','lradius','lwrist','lhand', 'rhumerus','rradius', 'rwrist','rhand']

    # initialize as tensor of zeros
    OptimJoints = []

    for i in range(len(children)):

        OptimJoints.append(torch.zeros(1, requires_grad=True))
        OptimJoints.append(torch.zeros(1, requires_grad=True))
        OptimJoints.append(torch.zeros(1, requires_grad=True))



    lr = 1e-1
    n_epochs = 100
    optimizer = optim.Adam(OptimJoints, lr=2)

    for epoch in range(n_epochs+1):


        if not epoch % 20:
            print(epoch)
            dummy_joints['root'].draw(goal_coord.detach().numpy())

        def closure():
            optimizer.zero_grad()


            est_pose = dummy_pose.copy()
            #loop and set new values
            for i in range(len(children)):
                # est_pose[children[i]] = OptimJoints[i]
                est_pose[children[i]] = torch.cat(OptimJoints[i*3:i*3+3])
            dummy_joints['root'].set_motion(est_pose)


            yhat = dummy_joints[goal_name].coordinate
            #yhat.requires_grad = True
            loss = torch.cdist(yhat.transpose(0, 1), y.transpose(0, 1), p=2.0)
            loss.backward(retain_graph=True)
            # print(loss)
            return loss
        optimizer.step(closure)
        with torch.no_grad():

            for i in range(len(children)):
                for lim in range(3):
                    OptimJoints[i * 3 + lim].clamp_(dummy_joints[children[i]].limits[lim][0],
                                                   dummy_joints[children[i]].limits[lim][1])
    print(OptimJoints)
    print(children)
if __name__ == '__main__':
    inverse_kinematics(goal_name, goal_coord, est_name, full_body = True)
