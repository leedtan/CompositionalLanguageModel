import re
import argparse
import random

parser = argparse.ArgumentParser(
    description='Script to generate and schedule navigation tasks')
parser.add_argument('--train_size', type=int, default=10000,
                    help='task instances shown during training')
args = parser.parse_args()

task_repository = {}
structure_repository = {}
task_set = set()

primitive_task_set = set()

command = "look"
primitive_task_set.add(command)
task_repository[command] = "I_LOOK"

command = "turn left"
primitive_task_set.add(command)
task_repository[command] = "I_TURN_LEFT"

command = "turn right"
primitive_task_set.add(command)
task_repository[command] = "I_TURN_RIGHT"

command = "walk"
primitive_task_set.add(command)
task_repository[command] = "I_WALK"

command = "run"
primitive_task_set.add(command)
task_repository[command] = "I_RUN"

command = "jump"
primitive_task_set.add(command)
task_repository[command] = "I_JUMP"

command = "look left"
primitive_task_set.add(command)
task_repository[command] = "I_TURN_LEFT I_LOOK"

command = "look right"
primitive_task_set.add(command)
task_repository[command] = "I_TURN_RIGHT I_LOOK"

command = "walk left"
primitive_task_set.add(command)
task_repository[command] = "I_TURN_LEFT I_WALK"

command = "walk right"
primitive_task_set.add(command)
task_repository[command] = "I_TURN_RIGHT I_WALK"

command = "run left"
primitive_task_set.add(command)
task_repository[command] = "I_TURN_LEFT I_RUN"

command = "run right"
primitive_task_set.add(command)
task_repository[command] = "I_TURN_RIGHT I_RUN"

command = "jump left"
primitive_task_set.add(command)
task_repository[command] = "I_TURN_LEFT I_JUMP"

command = "jump right"
primitive_task_set.add(command)
task_repository[command] = "I_TURN_RIGHT I_JUMP"

command = "turn around left"
primitive_task_set.add(command)
task_repository[command] = "I_TURN_LEFT I_TURN_LEFT I_TURN_LEFT I_TURN_LEFT"

command = "turn around right"
primitive_task_set.add(command)
task_repository[command] = "I_TURN_RIGHT I_TURN_RIGHT I_TURN_RIGHT I_TURN_RIGHT"

command = "look around left"
primitive_task_set.add(command)
task_repository[command] = "I_TURN_LEFT I_LOOK I_TURN_LEFT I_LOOK I_TURN_LEFT I_LOOK I_TURN_LEFT I_LOOK"

command = "look around right"
primitive_task_set.add(command)
task_repository[command] = "I_TURN_RIGHT I_LOOK I_TURN_RIGHT I_LOOK I_TURN_RIGHT I_LOOK I_TURN_RIGHT I_LOOK"

command = "walk around left"
primitive_task_set.add(command)
task_repository[command] = "I_TURN_LEFT I_WALK I_TURN_LEFT I_WALK I_TURN_LEFT I_WALK I_TURN_LEFT I_WALK"

command = "walk around right"
primitive_task_set.add(command)
task_repository[command] = "I_TURN_RIGHT I_WALK I_TURN_RIGHT I_WALK I_TURN_RIGHT I_WALK I_TURN_RIGHT I_WALK"

command = "run around left"
primitive_task_set.add(command)
task_repository[command] = "I_TURN_LEFT I_RUN I_TURN_LEFT I_RUN I_TURN_LEFT I_RUN I_TURN_LEFT I_RUN"

command = "run around right"
primitive_task_set.add(command)
task_repository[command] = "I_TURN_RIGHT I_RUN I_TURN_RIGHT I_RUN I_TURN_RIGHT I_RUN I_TURN_RIGHT I_RUN"

command = "jump around left"
primitive_task_set.add(command)
task_repository[command] = "I_TURN_LEFT I_JUMP I_TURN_LEFT I_JUMP I_TURN_LEFT I_JUMP I_TURN_LEFT I_JUMP"

command = "jump around right"
primitive_task_set.add(command)
task_repository[command] = "I_TURN_RIGHT I_JUMP I_TURN_RIGHT I_JUMP I_TURN_RIGHT I_JUMP I_TURN_RIGHT I_JUMP"

command = "turn opposite left"
primitive_task_set.add(command)
task_repository[command] = "I_TURN_LEFT I_TURN_LEFT"

command = "turn opposite right"
primitive_task_set.add(command)
task_repository[command] = "I_TURN_RIGHT I_TURN_RIGHT"

command = "look opposite left"
primitive_task_set.add(command)
task_repository[command] = "I_TURN_LEFT I_TURN_LEFT I_LOOK"

command = "look opposite right"
primitive_task_set.add(command)
task_repository[command] = "I_TURN_RIGHT I_TURN_RIGHT I_LOOK"

command = "walk opposite left"
primitive_task_set.add(command)
task_repository[command] = "I_TURN_LEFT I_TURN_LEFT I_WALK"

command = "walk opposite right"
primitive_task_set.add(command)
task_repository[command] = "I_TURN_RIGHT I_TURN_RIGHT I_WALK"

command = "run opposite left"
primitive_task_set.add(command)
task_repository[command] = "I_TURN_LEFT I_TURN_LEFT I_RUN"

command = "run opposite right"
primitive_task_set.add(command)
task_repository[command] = "I_TURN_RIGHT I_TURN_RIGHT I_RUN"

command = "jump opposite left"
primitive_task_set.add(command)
task_repository[command] = "I_TURN_LEFT I_TURN_LEFT I_JUMP"

command = "jump opposite right"
primitive_task_set.add(command)
task_repository[command] = "I_TURN_RIGHT I_TURN_RIGHT I_JUMP"

for k, v in task_repository.items():
    structure_repository[k] = [str(len(v.split(' ')))]

task_set = primitive_task_set

twice_task_set = set()

for primitive_command in task_set:
    command = primitive_command + " twice"
    twice_task_set.add(command)
    structure_repository[command] = structure_repository[primitive_command] * 2
    task_repository[command] = task_repository[primitive_command] + \
        " " + task_repository[primitive_command]

three_times_task_set = set()

for primitive_command in task_set:
    # command = primitive_command + " three times"
    command = primitive_command + " thrice"
    three_times_task_set.add(command)
    structure_repository[command] = structure_repository[primitive_command] * 3
    task_repository[command] = task_repository[primitive_command] + " " + \
        task_repository[primitive_command] + " " + \
        task_repository[primitive_command]

task_set = task_set | twice_task_set | three_times_task_set

and_task_set = set()

for primitive_command_1 in task_set:
    for primitive_command_2 in task_set:
        command = primitive_command_1 + " and " + primitive_command_2
        and_task_set.add(command)
        structure_repository[command] = structure_repository[primitive_command_1] + \
            structure_repository[primitive_command_2]
        task_repository[command] = task_repository[primitive_command_1] + \
            " " + task_repository[primitive_command_2]

after_task_set = set()

for primitive_command_1 in task_set:
    for primitive_command_2 in task_set:
        command = primitive_command_1 + " after " + primitive_command_2
        after_task_set.add(command)
        structure_repository[command] = structure_repository[primitive_command_2] + \
            structure_repository[primitive_command_1]
        task_repository[command] = task_repository[primitive_command_2] + \
            " " + task_repository[primitive_command_1]

task_set = task_set | and_task_set | after_task_set

# in_reverse_task_set = set()
# for primitive_command in task_set:
#     if (re.search(" ",task_repository[primitive_command])):
#         command = primitive_command + " in reverse"
#         in_reverse_task_set.add(command)
#         task_repository[command]= ' '.join(task_repository[primitive_command].split(' ')[::-1])
#
# task_set = task_set | in_reverse_task_set


for selected_task in task_repository:
    print("::: " + selected_task + " ::: " +
          task_repository[selected_task], " ::: ", ' '.join(structure_repository[selected_task]))

# for test_task in task_set:
#     print("CURRENT TEST TASK: " + test_task)
#     train_set = task_set - {test_task}
#     i=args.train_size
#     while (i>0):
#         selected_task = random.sample(train_set,1)[0]
#         print("IN: " + selected_task + " OUT: " + task_repository[selected_task])
#         i=i-1
#     print("IN: " + test_task + " OUT: " + task_repository[test_task])


# print("*** primitive *** : " + str(len(primitive_task_set)))
# for task in sorted(primitive_task_set):
#     print task + " " + task_repository[task]
#
# print("*** twice ***: " + str(len(twice_task_set)))
# for task in sorted(twice_task_set):
#     print task + " " + task_repository[task]
#
# print("*** and ***: " + str(len(and_task_set)))
# for task in sorted(and_task_set):
#     print task + " " + task_repository[task]
#
# print("*** reverse ***: " + str(len(in_reverse_task_set)))
# for task in sorted(in_reverse_task_set):
#     print task + " " + task_repository[task]
#
# print("*** all tasks ***: " + str(len(task_set)))
