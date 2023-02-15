from mobility.mobility_models import random_direction, reference_point_group
from gl_vars import gl


def set_mobility_model(SIM_SEED, frame_duration_s):
    # configures UE mobility
    if gl.UE_mobility_pattern == 'RDM':
        mobility_model = random_direction(gl.n_UEs, dimensions=(2 * gl.cell_radius_m, 2 * gl.cell_radius_m),
                                          height=gl.UE_height, velocity=(
                                          gl.UE_min_speed*frame_duration_s, gl.UE_average_speed*frame_duration_s),
                                          seed=SIM_SEED)
    elif gl.UE_mobility_pattern == 'RPGM':
        mobility_model = reference_point_group(gl.n_UEs, dimensions=(2 * gl.cell_radius_m, 2 * gl.cell_radius_m),
                                               height=gl.UE_height, velocity=(gl.UE_min_speed*frame_duration_s,
                                                                              gl.UE_average_speed*frame_duration_s),
                                               seed=SIM_SEED)
    elif gl.UE_mobility_pattern == 'stable':
        mobility_model = random_direction(gl.n_UEs, dimensions=(2 * gl.cell_radius_m, 2 * gl.cell_radius_m),
                                          height=gl.UE_height, velocity=(0, 0),
                                          seed=SIM_SEED)
    else:
        print("Currently only two mobility patterns are supported: RDM or RPGM."
              "Specify one of these two models.")
        assert ValueError

    return mobility_model
