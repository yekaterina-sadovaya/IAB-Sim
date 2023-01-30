from stat_container import st
from gl_vars import gl

import numpy as np
from gekko import GEKKO


class OptimizationParams(object):
    def __init__(self, ue_pos, ue_bs, ue_iab):
        self.lambda_b = gl.blockers_density
        self.B_backhaul = gl.bandwidth
        self.B_access = gl.bandwidth
        self.Delta = 1
        self.r_b = 0.3
        self.h_b = 1.7
        self.h_bs = 25
        self.h_iab = 15
        self.h_ue = 1.5
        self.P_bs = 40
        self.P_iab = 33
        self.f_c = 30e9
        self.cell_size = 300
        self.ue_bs = ue_bs
        self.ue_iab = ue_iab
        self.number_of_IAB = gl.n_IAB
        self.number_of_UE = gl.n_UEs
        self.bs_pos = gl.DgNB_pos
        self.iab_pos = gl.IAB_pos
        self.ue_pos = ue_pos


class BaseClass(object):
    def __init__(self, params):
        self.lambda_b = params.lambda_b
        self.number_of_IAB = params.number_of_IAB
        self.number_of_UE = params.number_of_UE
        self.B_backhaul = params.B_backhaul
        self.B_access = params.B_access
        self.Delta = params.Delta
        self.r_b = params.r_b
        self.h_b = params.h_b
        self.h_bs = params.h_bs
        self.h_iab = params.h_iab
        self.h_ue = params.h_ue
        self.P_bs = params.P_bs
        self.P_iab = params.P_iab
        self.f_c = params.f_c
        self.cell_size = params.cell_size
        self.bs_pos = params.bs_pos
        self.iab_pos = params.iab_pos
        self.ue_pos = params.ue_pos
        self.ue_bs = params.ue_bs
        self.ue_iab = params.ue_iab

    def optimizae_single_link(self, s_np_iab_DL, s_np_iab_UL, s_np_bs_DL, s_np_bs_UL, s_m_DL, s_m_UL):

        n_ue_iab = s_np_iab_DL.shape[0]
        n_ue_bs = s_np_bs_DL.shape[0]
        n_iab = self.iab_pos.shape[0]

        # Create model
        model = GEKKO()
        if gl.optimization_type == 'MAXMIN':
            # APOPT is an MINLP solver
            model.options.SOLVER = 1

        # Add variable z
        z = model.Var()

        # Add flow variables x_np for bs users
        # model.Var(lb=0, ub=0)
        y_1 = [model.Var(lb=0) for i in range(n_ue_bs)]
        y_2 = [model.Var(lb=0) for i in range(n_ue_bs)]
        y_3 = [model.Var(lb=0) for i in range(n_ue_bs)]
        y_4 = [model.Var(lb=0) for i in range(n_ue_bs)]

        # Add flow variables x_np for iab users
        x_1 = [model.Var(lb=0) for i in range(n_ue_iab)]
        x_2 = [model.Var(lb=0) for i in range(n_ue_iab)]
        x_3 = [model.Var(lb=0) for i in range(n_ue_iab)]
        x_4 = [model.Var(lb=0) for i in range(n_ue_iab)]

        # Add backhaul variables
        y_b1 = model.Var(lb=0)
        y_1b = model.Var(lb=0)

        # Add timeslot variables
        eps_1 = model.Var(lb=0)
        eps_2 = model.Var(lb=0)
        eps_3 = model.Var(lb=0)
        eps_4 = model.Var(lb=0)

        # Constraints
        if gl.optimization_type == 'MAXMIN':
            # Downlink
            for i in range(n_ue_bs):
                model.Equation(self.B_access * self.Delta * s_np_bs_DL[i] * (y_1[i] + y_3[i]) >= z)
            for i in range(n_ue_iab):
                model.Equation(self.B_access * self.Delta * s_np_iab_DL[i] * (x_3[i] + x_4[i]) >= z)

            # Uplink
            for i in range(n_ue_bs):
                model.Equation(self.B_access * self.Delta * s_np_bs_UL[i] * (y_2[i] + y_4[i]) >= z)
            for i in range(n_ue_iab):
                model.Equation(self.B_access * self.Delta * s_np_iab_UL[i] * (x_1[i] + x_2[i]) >= z)

            # Backhaul datarates
            for i in range(n_iab):
                model.Equation(sum(x*y for x, y in zip(s_np_iab_DL, x_3)) + sum(x*y for x, y in zip(s_np_iab_DL, x_4)) <= s_m_DL[0] * y_b1)
            for i in range(n_iab):
                model.Equation(sum(x*y for x, y in zip(s_np_iab_UL, x_1)) + sum(x*y for x, y in zip(s_np_iab_UL, x_2)) <= s_m_UL[0] * y_1b)

            # Timeslots constraints
            model.Equation(eps_1 + eps_2 + eps_3 + eps_4 == 1)

            model.Equation(model.sum(y_1) + y_b1 <= eps_1)
            model.Equation(model.sum(x_1) <= eps_1)
            model.Equation(model.sum(y_2) <= eps_2)
            model.Equation(model.sum(x_2) <= eps_2)
            model.Equation(model.sum(y_3) <= eps_3)
            model.Equation(model.sum(x_3) <= eps_3)
            model.Equation(model.sum(y_4) + y_1b <= eps_4)
            model.Equation(model.sum(x_4) <= eps_4)

            model.Obj(-z)  # Objective
            try:
                model.solve(disp=False)  # Solve
                y_1 = np.array(y_1)
                y_2 = np.array(y_2)
                y_3 = np.array(y_3)
                y_4 = np.array(y_4)
                x_1 = np.array(x_1)
                x_2 = np.array(x_2)
                x_3 = np.array(x_3)
                x_4 = np.array(x_4)
                y_b1 = np.array(y_b1)
                y_1b = np.array(y_1b)
                return y_1, y_2, y_3, y_4, x_1, x_2, x_3, x_4, y_b1, y_1b, eps_1, eps_2, eps_3, eps_4
            except:
                print('Optimization was not solved; Previous coefficients remain')
                return None

        elif gl.optimization_type == 'PF':
            # Backhaul datarates
            model.Equation(sum(x*y for x, y in zip(s_np_iab_DL, x_3)) + sum(x*y for x, y in zip(s_np_iab_DL, x_4)) <= s_m_DL[0] * y_b1)
            model.Equation(sum(x*y for x, y in zip(s_np_iab_UL, x_1)) + sum(x*y for x, y in zip(s_np_iab_UL, x_2)) <= s_m_UL[0] * y_1b)

            # Timeslots constraints
            model.Equation(eps_1 + eps_2 + eps_3 + eps_4 == 1)
            model.Equation(model.sum(y_1) + y_b1 <= eps_1)
            model.Equation(model.sum(x_1) <= eps_1)
            model.Equation(model.sum(y_2) <= eps_2)
            model.Equation(model.sum(x_2) <= eps_2)
            model.Equation(model.sum(y_3) <= eps_3)
            model.Equation(model.sum(x_3) <= eps_3)
            model.Equation(model.sum(y_4) + y_1b <= eps_4)
            model.Equation(model.sum(x_4) <= eps_4)

            # The problem objective to minimize
            downlink_donor = []
            uplink_donor = []
            for i in range(n_ue_bs):
                data_rate = model.log(self.B_access * self.Delta * s_np_bs_DL[i] * (y_1[i] + y_3[i]))
                downlink_donor.append(data_rate)
                data_rate = model.log(self.B_access * self.Delta * s_np_bs_UL[i] * (y_2[i] + y_4[i]))
                uplink_donor.append(data_rate)

            downlink_node = []
            uplink_node = []
            for i in range(n_ue_iab):
                data_rate = model.log(self.B_access * self.Delta * s_np_iab_DL[i] * (x_3[i] + x_4[i]))
                downlink_node.append(data_rate)
                data_rate = model.log(self.B_access * self.Delta * s_np_iab_UL[i] * (x_1[i] + x_2[i]))
                uplink_node.append(data_rate)

            model.Obj(-sum(downlink_donor) - sum(uplink_donor) -sum(downlink_node) - sum(uplink_node))
            try:
                model.solve(disp=True)  # Solve

                y_1 = np.array(y_1)
                y_2 = np.array(y_2)
                y_3 = np.array(y_3)
                y_4 = np.array(y_4)
                x_1 = np.array(x_1)
                x_2 = np.array(x_2)
                x_3 = np.array(x_3)
                x_4 = np.array(x_4)
                y_b1 = np.array(y_b1)
                y_1b = np.array(y_1b)
            except:
                print('Optimization was not solved; Previous coefficients remain')
                return None

            return y_1, y_2, y_3, y_4, x_1, x_2, x_3, x_4, y_b1, y_1b, eps_1, eps_2, eps_3, eps_4


    def post_process_results(self, y_1, y_2, y_3, y_4, x_1, x_2, x_3, x_4, y_b1, y_1b,
                             s_np_iab_DL, s_np_iab_UL, s_np_bs_DL, s_np_bs_UL, eps_1, eps_2, eps_3, eps_4):

        from iab_optimization.lib.plot_utilities import matplotlib_nikita_style
        matplotlib_nikita_style()

        n_ue_iab = s_np_iab_DL.shape[0]
        n_ue_bs = s_np_bs_DL.shape[0]

        # data rates
        h_DL_bs = np.empty((0, n_ue_bs))
        for i in range(n_ue_bs):
            h = self.B_access * self.Delta * s_np_bs_DL[i] * (y_1[i] + y_3[i])
            h_DL_bs = np.append(h_DL_bs, h)

        h_DL_iab = np.empty((0, n_ue_iab))
        for i in range(n_ue_iab):
            h = self.B_access * self.Delta * s_np_iab_DL[i] * (x_3[i] + x_4[i])
            h_DL_iab = np.append(h_DL_iab, h)

        h_UL_bs = np.empty((0, n_ue_bs))
        for i in range(n_ue_bs):
            h = self.B_access * self.Delta * s_np_bs_UL[i] * (y_2[i] + y_4[i])
            h_UL_bs = np.append(h_UL_bs, h)

        h_UL_iab = np.empty((0, n_ue_iab))
        for i in range(n_ue_iab):
            h = self.B_access * self.Delta * s_np_iab_UL[i] * (x_1[i] + x_2[i])
            h_UL_iab = np.append(h_UL_iab, h)

        h_DL_bs = h_DL_bs / 1e6
        h_DL_iab = h_DL_iab / 1e6
        h_UL_bs = h_UL_bs / 1e6
        h_UL_iab = h_UL_iab / 1e6

        # Timeslots at IAB-donor
        y = np.hstack((y_1, y_2, y_3, y_4))
        if len(y) != 0:
            y = y.sum(axis=0)
        backhaul = np.hstack((y_b1, 0, 0, y_1b))

        # Timeslots at IAB-node
        x = np.hstack((x_1, x_2, x_3, x_4))
        if len(x) != 0:
            x = x.sum(axis=0)

        eps = np.hstack((eps_1, eps_2, eps_3, eps_4))

        # h = np.hstack((h_DL_bs, h_DL_iab, h_UL_bs, h_UL_iab))
        h = np.array([])
        h = np.append(h, h_DL_bs)
        h = np.append(h, h_DL_iab)
        h = np.append(h, h_UL_bs)
        h = np.append(h, h_UL_iab)

        return h, eps, y, x, backhaul, [y_1, y_2, y_3, y_4], [x_1, x_2, x_3, x_4]


