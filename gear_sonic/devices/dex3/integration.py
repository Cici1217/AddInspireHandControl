"""Keep upstream Dex3 G1 creation and simulation behavior behind the common hook."""


def create_model(deployment):
    from gear_sonic.data.robot_model.instantiation.g1 import instantiate_g1_robot_model
    model = instantiate_g1_robot_model(
        waist_location=deployment.options.get('waist_location', 'lower_body'),
        high_elbow_pose=deployment.options.get('high_elbow_pose', False),
    )
    # Upstream supplemental limits repeat the left-hand signs on the right.
    # Keep body29 and upstream code intact; use the physical URDF limits here.
    for rows in deployment.registration.package['model']['joint_groups'].values():
        for row in rows:
            index = model.dof_index(row['urdf'])
            model.lower_joint_limits[index], model.upper_joint_limits[index] = row['limits']
            model.supplemental_info.joint_limits[row['urdf']] = list(row['limits'])
    return model


class Dex3SonicIntegration:
    def __init__(self, deployment):
        self.config = deployment

    def body_model(self):
        return self.config.robot_model()

    def simulation(self, wbc_config, launch):
        if launch.enable_depth_publish:
            raise ValueError('this device has no depth simulation publisher')
        return dict(wbc_config), {}

    def body_preflight(self, args):
        if args.remote or args.config or args.driver:
            raise ValueError('Dex3 body simulation does not use an external hand driver/configuration')

    def run_body(self, args, command, *, lease=None):
        from gear_sonic.scripts.run_device_body import run_process
        return run_process(command, lease)
