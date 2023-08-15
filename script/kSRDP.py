"""
Integer Programming method that solves kSRDP exact
"""

# import
import time
import logging
import numpy as np
import gurobipy as grb
import itertools as itl


# IP model
def kSRDP(name, adjacency, attack):
    """
    model:
    k: number of attacks;
    m: adjacency matrix, number of rows;
    n: adjacency matrix, number of columns;
    """

    # Basic settings
    model = grb.Model()
    load_time = time.time()

    # Getting scale
    m = adjacency.shape[0]
    n = adjacency.shape[1]
    if m != n:
        print("ERROR: GRAPH DIMENSIONS DON'T MATCH.")
        return
    if attack > m:
        print("ERROR: TOO MANY ATTACKS.")
        return

    # Neighbors
    neighbors = {}
    for i in range(m):
        neighbors[i] = []
        for j in range(n):
            if adjacency[i, j] == 1 and i != j:
                neighbors[i].append(j)
            else:
                continue

    # Identifying possible attack patterns
    print("Recognizing patterns...")
    patterns = {}
    combinations = list(itl.combinations(range(m), attack))
    for i in range(len(combinations)):
        patterns[i] = combinations[i]
    num_patterns = len(patterns)
    print("Patterns loaded!")

    # Adding variables
    # x_i: number of troops stationed at province i;
    print("Adding variables...")
    print("x_i...")
    var_x = {}
    for i in range(0, m):
        var_x[i] = model.addVar(
            lb=0.0,
            ub=grb.GRB.INFINITY,
            vtype=grb.GRB.INTEGER,
            name="x_{}".format(i)
        )
    model.update()

    # y_k.i.j: under attack pattern k, number of FAs transferring to i from j;
    # z_i: whether there's an legion at j;
    print("y_k.i.j & z_i...")
    var_y = {}
    var_z = {}
    for i in range(m):
        var_z[i] = model.addVar(
            lb=0.0,
            ub=1.0,
            vtype=grb.GRB.BINARY,
            name="z_{}".format(i)
        )
        for j in neighbors[i]:
            for k in range(num_patterns):
                var_y[k, i, j] = model.addVar(
                    lb=0.0,
                    ub=1.0,
                    vtype=grb.GRB.BINARY,
                    name="y_{}_{}_{}".format(k, i, j)
                )
    model.update()

    # Adding objective
    print("Setting Objective")
    objective = grb.quicksum([
        var_x[i]
        for i in range(m)
    ])
    model.setObjective(objective, grb.GRB.MINIMIZE)

    # Adding constrains
    # Constraint 1:
    M = 5
    print("Loading constraints...")
    for i in range(m):
        # Constraint 1
        model.addConstr(
            lhs=grb.quicksum(
                [var_z[i], -1.0 * var_x[i]]
            ),
            sense=grb.GRB.LESS_EQUAL,
            rhs=0.0
        )

        # Constraint 2
        model.addConstr(
            lhs=grb.quicksum(
                [M * var_z[i], -1.0 * var_x[i]]
            ),
            sense=grb.GRB.GREATER_EQUAL,
            rhs=0.0
        )
    model.update()

    for k in range(num_patterns):

        if k == int(0.1 * int(num_patterns)):
            print("|-----               10%                           |")
        if k == int(0.2 * int(num_patterns)):
            print("|----------          20%                           |")
        if k == int(0.3 * int(num_patterns)):
            print("|---------------     30%                           |")
        if k == int(0.4 * int(num_patterns)):
            print("|--------------------40%                           |")
        if k == int(0.5 * int(num_patterns)):
            print("|--------------------50%--                         |")
        if k == int(0.6 * int(num_patterns)):
            print("|--------------------60%-------                    |")
        if k == int(0.7 * int(num_patterns)):
            print("|--------------------70%------------               |")
        if k == int(0.8 * int(num_patterns)):
            print("|--------------------80%-----------------          |")
        if k == int(0.9 * int(num_patterns)):
            print("|--------------------90%----------------------     |")
        if k == int(num_patterns) - 1:
            print("|-------------------100%---------------------------|")

        # Constraint 3:
        for i in patterns[k]:
            model.addConstr(
                lhs=grb.quicksum([
                    var_z[i],
                    grb.quicksum([
                        var_y[k, i, j]
                        for j in neighbors[i]
                    ])
                ]),
                sense=grb.GRB.GREATER_EQUAL,
                rhs=1.0
            )
        model.update()

        # Constraint 4:
        for j in range(m):
            model.addConstr(
                lhs=grb.quicksum([
                    var_z[j],
                    grb.quicksum([
                        var_y[k, i, j]
                        for i in neighbors[j]
                    ]),
                    -1.0 * var_x[j]
                ]),
                sense=grb.GRB.LESS_EQUAL,
                rhs=0.0
            )
        model.update()

    # solving
    file_name = "solutions/{}.txt".format(name)
    file = open(file_name, "w+")
    file.write("Attack Number k = {}\n".format(attack))
    load_time = time.time() - load_time
    model.setParam(grb.GRB.Param.LogToConsole, 0)
    model.setParam(grb.GRB.Param.LogFile, file_name)
    model.optimize()

    # printing to file
    model_status = model.status
    file.write("Solution status = {}\n".format(model_status))
    if model_status == grb.GRB.OPTIMAL:
        file.write("Optimal value = {}\n".format(model.ObjVal))
        file.write("Loading time = {}\n".format(load_time))
        file.write("Solving time = {}\n".format(model.RunTime))
        file.write("===============================================\n")
        file.write("Number of Patterns = {}\n".format(len(patterns)))
        file.write("Number of Variables = {}\n".format(model.NumVars))
        file.write("Number of Constraitns = {}\n".format(model.NumConstrs))
        for i in range(m):
            file.write("x[{}] = {}\n".format(i, var_x[i].X))
    file.close()

    return 0


# second stage model
def second_stage(vertices, pattern, neighbors, defense, stay):
    """
    checking the integrality of a given pattern
    """

    second_stage_slack = grb.Model("second stage model with slack variables")
    second_stage_slack.setParam("OutputFlag", False)
    second_stage_slack.setParam("DualReductions", 0)

    # add varibles
    var_y = {}
    var_v = {}
    for j in vertices:
        # v
        var_v[j] = second_stage_slack.addVar(
            lb=0.0,
            ub=grb.GRB.INFINITY,
            vtype=grb.GRB.CONTINUOUS,
            name="v_{}".format(j)
        )
        for i in vertices:
            # y
            var_y[i, j] = second_stage_slack.addVar(
                lb=0.0,
                ub=grb.GRB.INFINITY,
                vtype=grb.GRB.CONTINUOUS,
                name="y_{}_{}".format(i, j)
            )
    second_stage_slack.update()

    # objective ------------------------------------------------------
    objective = grb.quicksum([
        grb.quicksum([
            -1.0 * var_v[i]
            for i in vertices
        ]),
        grb.quicksum([
            var_y[i, j]
            for i in vertices
            for j in vertices
        ])
    ])
    second_stage_slack.setObjective(objective, grb.GRB.MINIMIZE)

    # constriants ------------------------------------------------------
    for i in pattern:
        second_stage_slack.addConstr(
            lhs=grb.quicksum([
                var_y[i, j]
                for j in neighbors[i]
            ]),
            sense=grb.GRB.EQUAL,
            rhs=1 - stay[i]
        )
    second_stage_slack.update()
    for j in vertices:
        second_stage_slack.addConstr(
            lhs=grb.quicksum([
                var_v[j],
                grb.quicksum([
                    var_y[i, j]
                    for i in neighbors[j]
                ])
            ]),
            sense=grb.GRB.EQUAL,
            rhs=defense[j] - stay[j]
        )
    second_stage_slack.update()

    # solve ------------------------------------------------------
    second_stage_slack.optimize()

    # return
    status = second_stage_slack.status
    if status != grb.GRB.OPTIMAL:
        return {
            'status': status
        }
    else:
        val_y = {}
        for key in var_y.keys():
            val_y[key] = var_y[key].X
        val_v = {}
        for key in var_v.keys():
            val_v[key] = var_v[key].X
        return {
            'status': status,
            'objective': second_stage_slack.ObjVal,
            'val_y': val_y,
            'val_v': val_v
        }


# relaxed second stage dual
def duality(vertices, pattern, neighbors, defense, stay):
    """
    relaxed second stage durality
    """
    second_stage_dual_relaxed = grb.Model("Relaxed second stage dual")
    second_stage_dual_relaxed.setParam("OutputFlag", False)
    second_stage_dual_relaxed.setParam("DualReductions", 0)
    second_stage_dual_relaxed.setParam("InfUnbdInfo", 1)

    # add variable --------------------------------------------------
    var_lambda = {}
    for i in pattern:
        var_lambda[i] = second_stage_dual_relaxed.addVar(
            lb=-1 * grb.GRB.INFINITY,
            ub=grb.GRB.INFINITY,
            vtype=grb.GRB.CONTINUOUS,
            name="lambda_{}".format(i)
        )
    second_stage_dual_relaxed.update()
    var_mu = {}
    for j in vertices:
        var_mu[j] = second_stage_dual_relaxed.addVar(
            lb=-1 * grb.GRB.INFINITY,
            ub=grb.GRB.INFINITY,
            vtype=grb.GRB.CONTINUOUS,
            name="mu_{}".format(j)
        )
    second_stage_dual_relaxed.update()

    # objective --------------------------------------------------
    objective = grb.quicksum([
        # lambda
        grb.quicksum([
            (1 - stay[i]) * var_lambda[i]
            for i in pattern
        ]),
        # mu
        grb.quicksum([
            (defense[j] - stay[j]) * var_mu[j]
            for j in vertices
        ]),
    ])
    second_stage_dual_relaxed.setObjective(objective, grb.GRB.MAXIMIZE)

    # add constraints --------------------------------------------
    for j in vertices:
        for i in neighbors[j]:
            # i in pattern
            if i in pattern:
                second_stage_dual_relaxed.addConstr(
                    lhs=grb.quicksum([
                        var_lambda[i],
                        var_mu[j]
                    ]),
                    sense=grb.GRB.LESS_EQUAL,
                    rhs=1.0
                )
            # i not in pattern
            else:
                second_stage_dual_relaxed.addConstr(
                    lhs=grb.quicksum([
                        var_mu[j]
                    ]),
                    sense=grb.GRB.LESS_EQUAL,
                    rhs=1.0
                )
    second_stage_dual_relaxed.update()
    for j in vertices:
        second_stage_dual_relaxed.addConstr(
            lhs=grb.quicksum([
                var_mu[j]
            ]),
            sense=grb.GRB.LESS_EQUAL,
            rhs=-1.0
        )

    # solving ----------------------------------
    second_stage_dual_relaxed.optimize()

    # outcomes ---------------------------------
    status = second_stage_dual_relaxed.status

    if status == grb.GRB.OPTIMAL:
        return {
            'status': status,
            'objective': second_stage_dual_relaxed.ObjVal
        }
    elif status == grb.GRB.INFEASIBLE:
        return {
            'status': status
        }
    elif status == grb.GRB.UNBOUNDED:
        ray = {
            'lambda': {},
            'mu': {},
            'pi': {}
        }
        for key in var_lambda.keys():
            ray['lambda'][key] = var_lambda[key].UnbdRay
        for key in var_mu.keys():
            ray['mu'][key] = var_mu[key].UnbdRay
        return {
            'status': status,
            'ray': ray
        }
    else:
        return {
            'status': status
        }


# l-shaped
def Lshaped_kSRDP(name, adjacency, attack):
    """
    solving kSRDP using Benders decomposition.
    """
    run_time = time.time()
    # logging
    logging.basicConfig(
        filename='solutions_Lshaped/Lshaped_log.log',
        filemode='w+',
        format='%(levelname)s - %(message)s',
        level=logging.INFO
    )

    logging.info(
        "======================================={}"
        "========================================".format(
            name
        )
    )

    # set of vertices, [0, 1, 2, ...]
    vertices = range(adjacency.shape[0])

    # Neighbors, indexed by vertices
    neighbors = {}
    for i in vertices:
        neighbors[i] = []
        for j in vertices:
            if adjacency[i, j] == 1 and i != j:
                neighbors[i].append(j)
            else:
                continue

    # identify attack patterns, indexed by [0, 1, 2, ...]
    print("Recognizing patterns...")
    patterns = {}
    combinations = list(itl.combinations(vertices, attack))
    for i in range(len(combinations)):
        patterns[i] = combinations[i]
    num_patterns = len(patterns)
    ind_patterns = range(num_patterns)
    print("Patterns loaded!")

    # ===================================================================

    # constructing first stage model
    print("COnstructing first stage model...")
    first_stage = grb.Model("first_stage")
    first_stage.setParam("OutputFlag", False)

    # add variables
    var_x = {}
    var_z = {}
    for i in vertices:
        var_x[i] = first_stage.addVar(
            lb=0.0,
            ub=grb.GRB.INFINITY,
            vtype=grb.GRB.INTEGER,
            name="x_{}".format(i)
        )
        var_z[i] = first_stage.addVar(
            lb=0.0,
            ub=1.0,
            vtype=grb.GRB.BINARY,
            name="z_{}".format(i)
        )
    first_stage.update()

    # objective
    objective = grb.quicksum([
        grb.quicksum([
            var_x[i]
            for i in vertices
        ])
    ])
    first_stage.setObjective(objective, grb.GRB.MINIMIZE)

    # constraints: z_i and x_i
    M = 100
    for i in vertices:
        first_stage.addConstr(
            lhs=grb.quicksum([
                var_z[i],
                -1 * var_x[i]
            ]),
            sense=grb.GRB.LESS_EQUAL,
            rhs=0
        )
        first_stage.addConstr(
            lhs=grb.quicksum([
                M * var_z[i],
                -1 * var_x[i]
            ]),
            sense=grb.GRB.GREATER_EQUAL,
            rhs=0
        )
    first_stage.update()

    # ==================================================================

    # start iteration
    iter = 0
    while True:

        logging.info("Iteration: {}\n".format(iter))
        print("Iteration: {}".format(iter))

        # solve the first stage
        logging.info("Solving first stage...\n")
        first_stage.optimize()

        # solution status
        if first_stage.status != grb.GRB.OPTIMAL:
            logging.info("ERROR: First stage returns status code {}!\n".format(
                first_stage.status
            ))
            return 1
        else:
            # first stage is fesible, record solutions
            defense = {}
            for key in var_x.keys():
                defense[key] = var_x[key].X
            stay = {}
            for key in var_z.keys():
                stay[key] = var_z[key].X
            logging.info("Defense: {}\n".format(defense))
            logging.info("Current kSRDP: {}\n".format(
                np.sum([list(defense.values())])
            ))

        # ------------------------------------------------

        # checking second stage integral feasibility
        logging.info("Checking feasibility for second stage...\n")
        second_stage_feasible = True
        for p in ind_patterns:
            # solve dual
            dual_result = duality(
                vertices, patterns[p], neighbors, defense, stay
            )
            # unbounded
            if dual_result['status'] == grb.GRB.UNBOUNDED:
                logging.info(
                    "    Pattern No. {}, adding feasibility cut...\n".format(
                        p
                    )
                )
                # set flag
                second_stage_feasible = False
                # record ray
                ray = dual_result['ray']
                # add feasibility cut
                first_stage.addConstr(
                    lhs=grb.quicksum([
                        grb.quicksum([
                            (1 - var_z[i]) * ray['lambda'][i]
                            for i in patterns[p]
                        ]),
                        grb.quicksum([
                            (var_x[j] - var_z[j]) * ray['mu'][j]
                            for j in vertices
                        ])
                    ]),
                    sense=grb.GRB.LESS_EQUAL,
                    rhs=0.0
                )
                first_stage.update()
                break
            # optimal
            elif dual_result['status'] == grb.GRB.OPTIMAL:
                '''
                primal_result = second_stage(
                    vertices, patterns[p], neighbors, defense, stay
                )
                if primal_result['status'] == grb.GRB.OPTIMAL:
                    val_y = {}
                    for key in primal_result['val_y'].keys():
                        if primal_result['val_y'][key] > 0.0:
                            val_y[key] = primal_result['val_y'][key]
                    logging.info("    Second stage y: {}\n".format(
                        val_y
                    ))
                    continue
                else:
                    logging.info(
                        "    ERROR: Second stage primal "
                        "return code {}!\n".format(primal_result['status'])
                    )
                    return 1
                '''
                continue
            # infeasible
            elif dual_result['status'] == grb.GRB.INFEASIBLE:
                second_stage_feasible = False
                logging.info("ERROR: Second stage dual infeasible!\n")
                return 1
            # something else
            else:
                second_stage_feasible = False
                logging.info(
                    "ERROR: Second stage dual "
                    "return code {}!\n".format(dual_result['status'])
                )
                return 1

        # after checking for all patterns
        if second_stage_feasible:
            optimal_solution = defense
            optimal_kSRDN = np.sum(list(defense.values()))
            break
        else:
            # feasibility cut is added, continue loop to solve the first stage.
            iter += 1
            continue
    run_time = time.time() - run_time

    # output
    file = open(
        "solutions_Lshaped/{}.txt".format(name),
        "w+"
    )

    file.write("Optimal solution:\n")
    for key in optimal_solution.keys():
        file.write("Vertex[{}]: {}\n".format(key, optimal_solution[key]))
    file.write("\nOptimal k-SRDN: {}\n".format(
        optimal_kSRDN
    ))
    file.write("Run time: {}\n".format(
        run_time
    ))
    file.write("Iterations: {}\n".format(
        iter
    ))

    logging.shutdown()

    return 0
