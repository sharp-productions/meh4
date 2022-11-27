import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import seaborn as sns

import hiive.mdptoolbox.example
from pprint import pprint

# Default Taxi problems
P, R = hiive.mdptoolbox.example.openai("Taxi-v3")
print(P.shape)
print(R.shape)

# Value Iteration
vi = hiive.mdptoolbox.mdp.ValueIteration(P, R, 0.95)
run_stats = vi.run()
print("\nValue Iteration Results:")
print(run_stats[-1]['Iteration'], " iterations")
print(run_stats[-1])
rewards = []
# max_values = []
for i in run_stats:
    rewards.append(i['Reward'])
    # max_values.append(i['Max V'])
# plot the data
fig = plt.figure()
ax = fig.add_subplot()
ax.plot(range(len(rewards)), rewards, color='tab:blue')
# ax.plot(range(len(rewards)), max_values, color='tab:orange')
ax.set_title("Reward Convergence Plot - Default\nValue Iteration")
ax.set_xlabel("Iteration")
ax.set_ylabel("Reward")
plt.savefig("taxi-default-rewards-vi")
# print(run_info[-1])
# pprint(run_info)
print(len(vi.policy))
# print(vi.V)
print("\n")

# plot the policy
# get the slice of policies corresponding to passenger in car heading to bottom right location
PASSENGER_LOCATION = 4 # in taxi
DESTINATION = 3 # Blue square (bottom right)
optimal_policies = []
for row in range(5):
    row_values = []
    for col in range(5):
        index = (((((row*5)+col)*5)+PASSENGER_LOCATION)*4)+DESTINATION
        row_values.append(vi.policy[index])
    optimal_policies.append(row_values)
for row in range(5):
    print(optimal_policies[row])

fig = plt.figure()
fig.add_subplot(2, 2, 1)
ax2 = sns.heatmap(optimal_policies, linewidth=0, annot=True, cmap="Accent_r", cbar=False)
ax2.set_title(f"Optimal Policy - Default map (Value Iteration)")
# ax2.set_xlabel("Player sum")
# ax2.set_ylabel("Dealer showing")
plt.tick_params(left = False, right = False , labelleft = False ,
                labelbottom = False, bottom = False)

# add a legend
legend_elements = [
    Patch(facecolor="grey", edgecolor="black", label="South"),
    Patch(facecolor="lightgreen", edgecolor="black", label="Drop Off"),
    Patch(facecolor="lightblue", edgecolor="black", label="East"),
    Patch(facecolor="orange", edgecolor="black", label="North"),
    Patch(facecolor="pink", edgecolor="black", label="Pick Up"),
    Patch(facecolor="yellow", edgecolor="black", label="West"),
]
ax2.legend(handles=legend_elements, bbox_to_anchor=(1.6, 1))
plt.savefig("taxi-default-optimal-policy-vi")

# # Policy Iteration
# pi = hiive.mdptoolbox.mdp.PolicyIteration(P, R, 0.95)
# run_stats = pi.run()
# print("\nPolicy Iteration Results:")
# print(run_stats[-1]['Iteration'], " iterations")
# print(run_stats[-1])
# rewards = []
# # max_values = []
# for i in run_stats:
#     rewards.append(i['Reward'])
#     # max_values.append(i['Max V'])
# # plot the data
# fig = plt.figure()
# ax = fig.add_subplot(1, 1, 1)
# ax.plot(range(len(rewards)), rewards, color='tab:blue')
# # ax.plot(range(len(rewards)), max_values, color='tab:orange')
# ax.set_title("Reward Convergence Plot - Default\nPolicy Iteration")
# ax.set_xlabel("Iteration")
# ax.set_ylabel("Reward")
# plt.savefig("taxi-default-rewards-pi")
# # print(run_info[-1])
# # pprint(run_info)
# # print(pi.policy)
# # print(pi.V)
# print("\n")

# # Q Learning
# ql = hiive.mdptoolbox.mdp.QLearning(P,R,0.95, n_iter=2000000, alpha_decay=0.9999, epsilon_decay=0.9999)
# run_stats = ql.run()
# print("\nQ Learning Iteration Results:")
# print(run_stats[-1]['Iteration'], " iterations")
# print(run_stats[-1])
# # rewards = []
# q5_max_values = []
# iterations = []
# # print(run_stats[0])
# # print(run_stats[-1])
# for i in run_stats:
#     if i["Iteration"] % 100 == 0:
#         # rewards.append(i['Reward'])
#         q5_max_values.append(i['Max V'])
#         iterations.append(i['Iteration'])
# # plot the data
# fig_q = plt.figure()
# ax_q = fig_q.add_subplot()
# ax_q.plot(iterations, q5_max_values, label="5x5", color='tab:blue')
# # ax.plot(range(len(rewards)), max_values, color='tab:orange')
# ax_q.set_title("Reward Convergence Plot - Q Learning")
# ax_q.set_xlabel("Iteration")
# ax_q.set_xlim(iterations[0], iterations[-1])
# ax_q.set_ylabel("Reward")
# # plt.savefig("taxi-default-rewards-ql")
# pprint(run_stats[-1])
# # print("Q Table:\n", ql.Q)
# # print(ql.policy)
# # print(ql.V)

# # Larger Taxi problems
# P, R = hiive.mdptoolbox.example.openai("Taxi-v3", map_name="6")
# print(P.shape)
# print(R.shape)

# # Value Iteration
# vi = hiive.mdptoolbox.mdp.ValueIteration(P, R, 0.95)
# run_stats = vi.run()
# print("\nValue Iteration Results:")
# print(run_stats[-1]['Iteration'], " iterations")
# print(run_stats[-1])
# rewards = []
# # max_values = []
# for i in run_stats:
#     rewards.append(i['Reward'])
#     # max_values.append(i['Max V'])
# # plot the data
# fig = plt.figure()
# ax = fig.add_subplot(1, 1, 1)
# ax.plot(range(len(rewards)), rewards, color='tab:blue')
# # ax.plot(range(len(rewards)), max_values, color='tab:orange')
# ax.set_title("Reward Convergence Plot - 6x6\nValue Iteration")
# ax.set_xlabel("Iteration")
# ax.set_ylabel("Reward")
# plt.savefig("taxi-6x6-rewards-vi")
# # print(run_info[-1])
# # pprint(run_info)
# # print(vi.policy)
# # print(vi.V)
# print("\n")

# # Policy Iteration
# pi = hiive.mdptoolbox.mdp.PolicyIteration(P, R, 0.95)
# run_stats = pi.run()
# print("\nPolicy Iteration Results:")
# print(run_stats[-1]['Iteration'], " iterations")
# print(run_stats[-1])
# rewards = []
# # max_values = []
# for i in run_stats:
#     rewards.append(i['Reward'])
#     # max_values.append(i['Max V'])
# # plot the data
# fig = plt.figure()
# ax = fig.add_subplot(1, 1, 1)
# ax.plot(range(len(rewards)), rewards, color='tab:blue')
# # ax.plot(range(len(rewards)), max_values, color='tab:orange')
# ax.set_title("Reward Convergence Plot - 6x6\nPolicy Iteration")
# ax.set_xlabel("Iteration")
# ax.set_ylabel("Reward")
# plt.savefig("taxi-6x6-rewards-pi")
# # print(run_info[-1])
# # pprint(run_info)
# # print(pi.policy)
# # print(pi.V)
# print("\n")

# # Q Learning
# ql = hiive.mdptoolbox.mdp.QLearning(P,R,0.95, n_iter=2000000, alpha_decay=0.9999, epsilon_decay=0.9999)
# run_stats = ql.run()
# print("\nQ Learning Iteration Results:")
# print(run_stats[-1]['Iteration'], " iterations")
# print(run_stats[-1])
# # rewards = []
# q6_max_values = []
# iterations = []
# # print(run_stats[0])
# # print(run_stats[-1])
# for i in run_stats:
#     if i["Iteration"] % 100 == 0:
#         # rewards.append(i['Reward'])
#         q6_max_values.append(i['Max V'])
#         iterations.append(i['Iteration'])
# # plot the data
# # ax = fig.add_subplot()
# ax_q.plot(iterations, q6_max_values, label="6x6", color='tab:orange')
# # ax.plot(range(len(rewards)), max_values, color='tab:orange')
# # ax_q.set_title("Reward Convergence Plot - 6x6\nQ Learning")
# # ax.set_xlabel("Iteration")
# # ax.set_xlim(iterations[0], iterations[-1])
# # ax.set_ylabel("Reward")
# # plt.savefig("taxi-6x6-rewards-ql")
# pprint(run_stats[-1])
# # print("Q Table:\n", ql.Q)
# # print(ql.policy)
# # print(ql.V)

# # Largerer Taxi problems
# P, R = hiive.mdptoolbox.example.openai("Taxi-v3", map_name="7")
# print(P.shape)
# print(R.shape)

# # Value Iteration
# vi = hiive.mdptoolbox.mdp.ValueIteration(P, R, 0.95)
# run_stats = vi.run()
# print("\nValue Iteration Results:")
# print(run_stats[-1]['Iteration'], " iterations")
# print(run_stats[-1])
# rewards = []
# # max_values = []
# for i in run_stats:
#     rewards.append(i['Reward'])
#     # max_values.append(i['Max V'])
# # plot the data
# fig = plt.figure()
# ax = fig.add_subplot(1, 1, 1)
# ax.plot(range(len(rewards)), rewards, color='tab:blue')
# # ax.plot(range(len(rewards)), max_values, color='tab:orange')
# ax.set_title("Reward Convergence Plot - 7x7\nValue Iteration")
# ax.set_xlabel("Iteration")
# ax.set_ylabel("Reward")
# plt.savefig("taxi-7x7-rewards-vi")
# # print(run_info[-1])
# # pprint(run_info)
# # print(vi.policy)
# # print(vi.V)
# print("\n")

# # Policy Iteration
# pi = hiive.mdptoolbox.mdp.PolicyIteration(P, R, 0.95)
# run_stats = pi.run()
# print("\nPolicy Iteration Results:")
# print(run_stats[-1]['Iteration'], " iterations")
# print(run_stats[-1])
# rewards = []
# # max_values = []
# for i in run_stats:
#     rewards.append(i['Reward'])
#     # max_values.append(i['Max V'])
# # plot the data
# fig = plt.figure()
# ax = fig.add_subplot(1, 1, 1)
# ax.plot(range(len(rewards)), rewards, color='tab:blue')
# # ax.plot(range(len(rewards)), max_values, color='tab:orange')
# ax.set_title("Reward Convergence Plot - 7x7\nPolicy Iteration")
# ax.set_xlabel("Iteration")
# ax.set_ylabel("Reward")
# plt.savefig("taxi-7x7-rewards-pi")
# # print(run_info[-1])
# # pprint(run_info)
# # print(pi.policy)
# # print(pi.V)
# print("\n")

# # Q Learning
# ql = hiive.mdptoolbox.mdp.QLearning(P,R,0.95, n_iter=2000000, alpha_decay=0.9999, epsilon_decay=0.9999)
# run_stats = ql.run()
# print("\nQ Learning Iteration Results:")
# print(run_stats[-1]['Iteration'], " iterations")
# print(run_stats[-1])
# # rewards = []
# q7_max_values = []
# iterations = []
# # print(run_stats[0])
# # print(run_stats[-1])
# for i in run_stats:
#     if i["Iteration"] % 100 == 0:
#         # rewards.append(i['Reward'])
#         q7_max_values.append(i['Max V'])
#         iterations.append(i['Iteration'])
# # plot the data
# # fig = plt.figure()
# # ax_q = fig_q.add_subplot()
# ax_q.plot(iterations, q7_max_values, label="7x7", color='tab:green')
# # ax.plot(range(len(rewards)), max_values, color='tab:orange')
# ax_q.set_title("Reward Convergence Plot - 7x7\nQ Learning")
# ax_q.set_xlabel("Iteration")
# ax_q.set_xlim(iterations[0], iterations[-1])
# ax_q.set_ylabel("Reward")
# ax_q.legend(title='Grid Size')
# plt.savefig("taxi-rewards-ql")
# pprint(run_stats[-1])
# # print("Q Table:\n", ql.Q)
# # print(ql.policy)
# # print(ql.V)


# # Forest problems
# P, R = hiive.mdptoolbox.example.forest(S=3, r1=4, r2=2, p=0.05)

# vi = hiive.mdptoolbox.mdp.ValueIteration(P, R, 0.95)
# run_info = vi.run()
# print("\nValue Iteration Results:")
# print(len(run_info), " iterations")
# # print(run_info[-1])
# # pprint(run_info)
# print(vi.policy)
# print(vi.V)
# print("\n")

# pi = hiive.mdptoolbox.mdp.PolicyIteration(P, R, 0.95)
# run_info = pi.run()
# print("\nPolicy Iteration Results:")
# print(len(run_info), " iterations")
# # print(run_info[-1])
# # pprint(run_info)
# print(pi.policy)
# print(pi.V)
# print("\n")

# ql = hiive.mdptoolbox.mdp.QLearning(P,R,0.95, n_iter=300000)
# run_info = ql.run()
# print("\nQLearning Iteration Results:")
# pprint(run_info[-1])
# print("Q Table:\n", ql.Q)
# print(ql.policy)
# print(ql.V)