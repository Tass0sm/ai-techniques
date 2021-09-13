extern crate nalgebra as na;

fn main() {
    let axis  = Vector3::x_axis();
    let angle = 1.57;
    let b     = Rotation3::from_axis_angle(&axis, angle);

    relative_eq!(b.axis().unwrap(), axis);
    relative_eq!(b.angle(), angle);
}
