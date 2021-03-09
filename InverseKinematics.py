from dummy import *
import torch.optim as optim
import torch
torch.manual_seed(1510)

est_name = 'rhumerus'
goal_name = 'rwrist'



dummy_joints, dummy_pose = dummy()

goal_pose = dummy_pose.copy()
goal_pose[est_name] = torch.tensor([90, 0, 0])


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

def inverse_kinematics(goal_name, goal_coord,est_name):
    dummy_joints, dummy_pose = dummy()
    dummy_joints['root'].set_motion(dummy_pose)
    y = goal_coord
    #L = torch.cdist(yhat.transpose(0,1), y.transpose(0,1), p=2.0)
    #yhat = dummy_joints[goal_name].coordinate

    #fill children list with children
    children = getChildren(est_name, goal_name, childrenlist = list())

    # initialize as tensor of zeros
    OptimJoints = []

    for i in range(len(children)):
        a = torch.zeros(3, requires_grad=True)
        #for lim in range(3):
        #    a = a.clamp(dummy_joints[children[i]].limits[lim][0],dummy_joints[children[i]].limits[lim][1])
        OptimJoints.append(a)
    # for _ in dummy_joints[est_name].children:
        #tilføj clamps her
        # a = torch.zeros(3, requires_grad=True)
        # OptimJoints.append(a)



    lr = 1e-1
    n_epochs = 100
    optimizer = optim.LBFGS(OptimJoints, lr=1)
    for epoch in range(n_epochs+1):
        if not epoch % 20:
            print(epoch)
            dummy_joints['root'].draw(goal_coord.detach().numpy())

        def closure():
            optimizer.zero_grad()
            est_pose = dummy_pose.copy()
            #loop and set new values
            for i in range(len(children)):
                est_pose[children[i]] = OptimJoints[i]

            dummy_joints['root'].set_motion(est_pose)


            yhat = dummy_joints[goal_name].coordinate
            #yhat.requires_grad = True
            loss = torch.cdist(yhat.transpose(0, 1), y.transpose(0, 1), p=2.0)
            loss.backward(retain_graph=True)
            return loss
        optimizer.step(closure)
    print(OptimJoints)

if __name__ == '__main__':
    inverse_kinematics(goal_name, goal_coord, est_name)