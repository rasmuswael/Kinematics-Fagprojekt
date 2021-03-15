import torch
from pyTorch_parser import Joint, read_line


def dummy():
    # Read 01.asf (placeholder) to get rudementary information, but set all directions to zero.
    file_path = './data/01/01.asf'
    '''read joint data only'''
    with open(file_path) as f:
        content = f.read().splitlines()  # splits a string into a list. The splitting is done at line breaks

    for idx, line in enumerate(content):
        # meta infomation is ignored
        if line == ':bonedata':
            content = content[idx + 1:]
            break

    # read joints
    joints = {'root': Joint('root', torch.zeros(3), 0, torch.zeros(3), [], [])}
    idx = 0
    while True:
        # the order of each section is hard-coded

        line, idx = read_line(content, idx)

        if line[0] == ':hierarchy':
            break

        assert line[0] == 'begin'

        line, idx = read_line(content, idx)
        assert line[0] == 'id'

        line, idx = read_line(content, idx)
        assert line[0] == 'name'
        name = line[1]

        line, idx = read_line(content, idx)
        assert line[0] == 'direction'
        direction = torch.tensor([float(axis) for axis in line[1:]])

        # skip length
        line, idx = read_line(content, idx)
        assert line[0] == 'length'
        length = float(line[1])

        line, idx = read_line(content, idx)
        assert line[0] == 'axis'
        assert line[4] == 'XYZ'

        # set axis to 0
        # axis = torch.tensor([float(axis) for axis in line[1:-1]])
        axis = torch.tensor([float(axis) for axis in line[1:-1]])

        dof = []
        limits = []

        line, idx = read_line(content, idx)
        if line[0] == 'dof':
            dof = line[1:]
            for i in range(len(dof)):
                line, idx = read_line(content, idx)
                if i == 0:
                    assert line[0] == 'limits'
                    line = line[1:]
                assert len(line) == 2
                mini = float(line[0][1:])
                maxi = float(line[1][:-1])
                limits.append((mini, maxi))

            line, idx = read_line(content, idx)

        assert line[0] == 'end'
        joints[name] = Joint(
            name,
            direction,
            length,
            axis,
            dof,
            limits
        )

    # read hierarchy
    assert line[0] == ':hierarchy'

    line, idx = read_line(content, idx)

    assert line[0] == 'begin'

    while True:
        line, idx = read_line(content, idx)
        if line[0] == 'end':
            break
        assert len(line) >= 2
        for joint_name in line[1:]:
            joints[line[0]].children.append(joints[joint_name])
        for nm in line[1:]:
            joints[nm].parent = joints[line[0]]

    #Def t-pose
    file_path = './data/01/01_01.amc'
    with open(file_path) as f:
        content = f.read().splitlines()

    for idx, line in enumerate(content):
        if line == ':DEGREES':
            content = content[idx + 1:]
            break

    joint_names = []
    idx = 0
    line, idx = read_line(content, idx)
    assert line[0].isnumeric(), line

    while True:
        line, idx = read_line(content, idx)
        if line is None:
            EOF = True
            break
        if line[0].isnumeric():
            break
        joint_names.append(line[0])

    pose = {'root': torch.tensor(6 * [float(0)], requires_grad=False)}
    for name in joint_names:
        if name != 'root':
            numaxis = 0
            for axis, lm in enumerate(joints[name].limits):
                if not torch.equal(torch.tensor(lm).float(), torch.zeros(2)):
                    numaxis += 1
            pose[name] = torch.tensor(numaxis * [float(0)], requires_grad=False)
    return joints, pose

if __name__ == '__main__':
    dummy_joints, dummy_pose = dummy()
    dummy_joints['root'].set_motion(dummy_pose)
    dummy_joints['root'].draw()