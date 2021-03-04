from dummy import *
import torch.optim as optim
torch.manual_seed(1510)

def loss(y, yhat, type):
    if type == 'euclidian':
        L = torch.cdist(yhat.transpose(0,1), y.transpose(0,1), p=2.0)
        return L

def simple_experiment():
    dummy_joints, dummy_pose = dummy()
    goal_pose = dummy_pose.copy()
    est_name = 'rhumerus'
    goal_name = 'rwrist'
    goal_pose[est_name] = torch.tensor([90,0,0])

    dummy_joints['root'].set_motion(dummy_pose)
    # dummy_joints['root'].draw()

    yhat = dummy_joints[goal_name].coordinate

    dummy_joints['root'].set_motion(goal_pose)
    # dummy_joints['root'].draw()

    y = dummy_joints[goal_name].coordinate

    L = torch.cdist(yhat.transpose(0,1), y.transpose(0,1), p=2.0)
    # print(sum([i**2 for i in (y - yhat)])**(1/2))

    # initialize as tensor of zeros
    a = torch.zeros(3, requires_grad=True)

    # lr = 1e-1
    # n_epochs = 1000
    # optimizer = optim.Adam([a], lr=lr)
    #
    # for epoch in range(n_epochs):
    #     est_pose = dummy_pose.copy()
    #     est_pose[est_name] = a
    #     dummy_joints['root'].set_motion(est_pose)
    #     yhat = dummy_joints[goal_name].coordinate
    #     loss = torch.cdist(yhat.transpose(0,1), y.transpose(0,1), p=2.0)
    #     loss.backward(retain_graph=True)
    #     optimizer.step(closure=None)
    #     optimizer.zero_grad()
    lr = 1e-1
    n_epochs = 50
    optimizer = optim.LBFGS([a], lr=1)
    for epoch in range(n_epochs):
        if not epoch % 10:
            print(epoch)
        def closure():
            optimizer.zero_grad()
            est_pose = dummy_pose.copy()
            est_pose[est_name] = a
            dummy_joints['root'].set_motion(est_pose)
            yhat = dummy_joints[goal_name].coordinate
            loss = torch.cdist(yhat.transpose(0, 1), y.transpose(0, 1), p=2.0)
            loss.backward(retain_graph=True)
            return loss
        optimizer.step(closure)
    print(a)

if __name__ == '__main__':
    simple_experiment()