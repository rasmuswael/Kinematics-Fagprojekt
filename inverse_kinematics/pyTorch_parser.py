#Copyright (c) 2018 Yuxiao Zhou

import numpy as np
import torch
import matplotlib.pyplot as plt
#from transforms3d.euler import euler2mat
from pytorch3d.transforms import euler_angles_to_matrix
from mpl_toolkits.mplot3d import Axes3D

class Joint:
  def __init__(self, name, direction, length, axis, dof, limits):
    """
    Definition of basic joint. The joint also contains the information of the
    bone between it's parent joint and itself. Refer
    [here](https://research.cs.wisc.edu/graphics/Courses/cs-838-1999/Jeff/ASF-AMC.html)
    for detailed description for asf files.

    Parameter
    ---------
    name: Name of the joint defined in the asf file. There should always be one
    root joint. String.

    direction: Default direction of the joint(bone). The motions are all defined
    based on this default pose.

    length: Length of the bone.

    axis: Axis of rotation for the bone.

    dof: Degree of freedom. Specifies the number of motion channels and in what
    order they appear in the AMC file.

    limits: Limits on each of the channels in the dof specification

    """
    self.name = name
    self.direction = torch.reshape(direction, [3, 1]) #from 1x3 to 3x1 array
    self.length = length
    axis = torch.deg2rad(axis).flip(0) #Convert angles from degrees to radians
    self.C = euler_angles_to_matrix(axis, "ZYX") #return rotation matrix from Euler angles and axis sequence.
    # self.Cinv = torch.inverse(self.C)
    self.Cinv = torch.transpose(self.C, 0, 1)
    self.limits = np.zeros([3, 2])
    for lm, nm in zip(limits, dof):
      if nm == 'rx':
        self.limits[0] = lm
      elif nm == 'ry':
        self.limits[1] = lm
      else:
        self.limits[2] = lm
    self.parent = None
    self.children = []
    self.coordinate = None
    self.matrix = None

  def set_motion(self, motion):
    if self.name == 'root':
      self.coordinate = torch.reshape(motion['root'][:3], [3, 1]) #
      rotation = torch.deg2rad(motion['root'][3:]).flip(0)
      # self.matrix = self.C.dot(euler_angles_to_matrix(rotation, "XYZ")).dot(self.Cinv)
      # print(euler_angles_to_matrix(rotation, "ZYX"))
      self.matrix = torch.mm(torch.mm(self.C, euler_angles_to_matrix(rotation, "ZYX")), self.Cinv)
    else:
      idx = 0
      rotation = torch.zeros(3)
      for axis, lm in enumerate(self.limits):
        if not torch.equal(torch.tensor(lm).float(), torch.zeros(2)):
          rotation[axis] = motion[self.name][idx]
          idx += 1
      rotation = torch.deg2rad(rotation).flip(0)
      # self.matrix = self.parent.matrix.dot(self.C).dot(euler_angles_to_matrix(rotation, "XYZ")).dot(self.Cinv)
      self.matrix = torch.mm(torch.mm(torch.mm(self.parent.matrix, self.C), euler_angles_to_matrix(rotation, "ZYX")), self.Cinv)
      # self.coordinate = self.parent.coordinate + self.length * self.matrix.dot(self.direction)
      self.coordinate = self.parent.coordinate + self.length * torch.mm(self.matrix, self.direction)
    for child in self.children:
      child.set_motion(motion)

  def draw(self, *args):
    joints = self.to_dict()
    fig = plt.figure()
    ax = Axes3D(fig)

    # Center around origo
    ax.set_xlim3d(-30, 30)
    ax.set_ylim3d(-30, 30)
    ax.set_zlim3d(-20, 40)

    xs, ys, zs = [], [], []
    for joint in joints.values():
      joint.coordinate = joint.coordinate.detach().numpy()
      xs.append(joint.coordinate[0, 0])
      ys.append(joint.coordinate[1, 0])
      zs.append(joint.coordinate[2, 0])
    plt.plot(zs, xs, ys, 'b.')

    for joint in joints.values():
      child = joint
      if child.parent is not None:
        parent = child.parent
        # child.coordinate, parent.coordinate = child.coordinate.detach().numpy(), parent.coordinate.detach().numpy()
        xs = [child.coordinate[0, 0], parent.coordinate[0, 0]]
        ys = [child.coordinate[1, 0], parent.coordinate[1, 0]]
        zs = [child.coordinate[2, 0], parent.coordinate[2, 0]]
        plt.plot(zs, xs, ys, 'r')
    if args != ():
      ax.plot(args[0][2], args[0][0], args[0][1], markerfacecolor='k', markeredgecolor='k', marker='o',
              markersize=5, alpha=1)
    plt.show()

  def to_dict(self):
    ret = {self.name: self}
    for child in self.children:
      ret.update(child.to_dict())
    return ret

  def pretty_print(self):
    print('===================================')
    print('joint: %s' % self.name)
    print('direction:')
    print(self.direction)
    print('limits:', self.limits)
    print('parent:', self.parent)
    print('children:', self.children)


def read_line(stream, idx):
  if idx >= len(stream):
    return None, idx
  line = stream[idx].strip().split()
  idx += 1
  return line, idx


def parse_asf(file_path):
  '''read joint data only'''
  with open(file_path) as f:
    content = f.read().splitlines() # splits a string into a list. The splitting is done at line breaks

  for idx, line in enumerate(content):
    # meta infomation is ignored
    if line == ':bonedata':
      content = content[idx+1:]
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

  return joints


def parse_amc(file_path, ignore_translation = True):
  with open(file_path) as f:
    content = f.read().splitlines()

  for idx, line in enumerate(content):
    if line == ':DEGREES':
      content = content[idx+1:]
      break

  frames = []
  idx = 0
  line, idx = read_line(content, idx)
  assert line[0].isnumeric(), line
  EOF = False
  while not EOF:
    joint_degree = {}
    while True:
      line, idx = read_line(content, idx)
      if line is None:
        EOF = True
        break
      if line[0].isnumeric():
        break

      # Ignore translations
      if line[0] == 'root' and ignore_translation:
        joint_degree[line[0]] = torch.tensor([0,0,0] + [float(deg) for deg in line[4:]], requires_grad=False)
      else:
        joint_degree[line[0]] = torch.tensor([float(deg) for deg in line[1:]], requires_grad=False)
      # joint_degree[line[0]] = torch.tensor([float(deg) for deg in line[1:]], requires_grad=False)
    frames.append(joint_degree)
  return frames


def test_all():
  import os
  lv0 = './data'
  lv1s = os.listdir(lv0)
  for lv1 in lv1s:
    lv2s = os.listdir('/'.join([lv0, lv1]))
    asf_path = '%s/%s/%s.asf' % (lv0, lv1, lv1)
    print('parsing %s' % asf_path)
    joints = parse_asf(asf_path)
    motions = parse_amc('%s/%s/%s_%s.amc' % (lv0, lv1, lv1, lv1))
    joints['root'].set_motion(motions[0])
    joints['root'].draw()

    # for lv2 in lv2s:
    #   if lv2.split('.')[-1] != 'amc':
    #     continue
    #   amc_path = '%s/%s/%s' % (lv0, lv1, lv2)
    #   print('parsing amc %s' % amc_path)
    #   motions = parse_amc(amc_path)
    #   for idx, motion in enumerate(motions):
    #     print('setting motion %d' % idx)
    #     joints['root'].set_motion(motion)


if __name__ == '__main__':
  # test_all()
  asf_path = './data/123/123.asf'
  amc_path = './data/123/123_01.amc'
  joints = parse_asf(asf_path)
  motions = parse_amc(amc_path)
  frame_idx = 0
  joints['root'].set_motion(motions[frame_idx])
  joints['root'].draw()
