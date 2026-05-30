//! Rapier physics world wrapper. Owns every set/pipeline Rapier needs and runs
//! one fixed-timestep step at a time.

use rapier3d::prelude::*;

/// The fixed physics timestep (seconds). The render loop feeds real time into an
/// accumulator and calls [`Physics::step`] once per elapsed slice of this size.
pub const FIXED_DT: f32 = 1.0 / 60.0;

pub struct Physics {
    pub gravity: Vector<Real>,
    pub integration_parameters: IntegrationParameters,
    pub islands: IslandManager,
    pub broad_phase: DefaultBroadPhase,
    pub narrow_phase: NarrowPhase,
    pub bodies: RigidBodySet,
    pub colliders: ColliderSet,
    pub impulse_joints: ImpulseJointSet,
    pub multibody_joints: MultibodyJointSet,
    pub ccd_solver: CCDSolver,
    pub query_pipeline: QueryPipeline,
    pub physics_pipeline: PhysicsPipeline,
}

impl Physics {
    pub fn new() -> Self {
        let mut integration_parameters = IntegrationParameters::default();
        integration_parameters.dt = FIXED_DT;
        Self {
            // Stronger-than-real gravity gives a planted, less "floaty" arcade feel.
            gravity: vector![0.0, -20.0, 0.0],
            integration_parameters,
            islands: IslandManager::new(),
            broad_phase: DefaultBroadPhase::new(),
            narrow_phase: NarrowPhase::new(),
            bodies: RigidBodySet::new(),
            colliders: ColliderSet::new(),
            impulse_joints: ImpulseJointSet::new(),
            multibody_joints: MultibodyJointSet::new(),
            ccd_solver: CCDSolver::new(),
            query_pipeline: QueryPipeline::new(),
            physics_pipeline: PhysicsPipeline::new(),
        }
    }

    /// Advance the simulation by exactly one [`FIXED_DT`] slice.
    pub fn step(&mut self) {
        self.physics_pipeline.step(
            &self.gravity,
            &self.integration_parameters,
            &mut self.islands,
            &mut self.broad_phase,
            &mut self.narrow_phase,
            &mut self.bodies,
            &mut self.colliders,
            &mut self.impulse_joints,
            &mut self.multibody_joints,
            &mut self.ccd_solver,
            Some(&mut self.query_pipeline),
            &(),
            &(),
        );
    }
}
