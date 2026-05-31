import numpy as np
import porepy as pp


class PreinjectionStabilization:
    """Check that the displacement field has stabilized before injection starts.

    Compares displacement at two times: just before ``stabilization_time`` and
    at the stabilization_time. If the relative change is below the tolerance, the model is
    considered stabilized.
    """

    def calculate_time_to_stabilization(self, tolerance) -> None:
        """
        Args:
            tolerance: Maximum allowed relative change in displacement
                (||u_1 - u_0|| / ||u_1||) for the model to count as stabilized.

        Returns:
            None. Prints a message saying whether stabilization was reached.
        """
        # The time at which we want the model to be stabilized.
        dt = self.params["stabilization_time"]

        # Before the target time: keep overwriting displacement_0 so it holds
        # the latest pre-target displacement when we cross dt. The
        # self.time_manager.time > 0 guard skips the initial step.
        if self.time_manager.time < dt and self.time_manager.time > 0:
            self.displacement_0 = self.evaluate_and_scale(self.mdg.subdomains(dim=self.nd), "displacement", "m")

        # First step at or after the target time: take the second snapshot and
        # compare. Runs only once thanks to the hasattr guard.
        if self.time_manager.time >= dt and not hasattr(self, "displacement_1"):
            self.displacement_1 = self.evaluate_and_scale(self.mdg.subdomains(dim=self.nd), "displacement", "m")
            change_displacement = self.displacement_1 - self.displacement_0
            tot_change = np.linalg.norm(change_displacement)
            tot_displacement_1 = np.linalg.norm(self.displacement_1)
            relative_change = tot_change / tot_displacement_1 if tot_displacement_1 > 0 else 0.0

            if relative_change <= tolerance:
                print(f"Stabilization achieved at stabilization time {dt/pp.YEAR:.0f} years with relative change {relative_change:.3e}.")
            else:
                print(f"Not yet stabilized, {relative_change:.3e}")

class ScheduleClippingTimeManager(pp.TimeManager):
    """TimeManager that fix time iteration for the schedule. Written by claude
    """

    def compute_time_step(self, *args, **kwargs):
        dt = super().compute_time_step(*args, **kwargs)
        if dt is None:
            return None
        if self._scheduled_idx < len(self.schedule):
            # Find next target time and remaining time of this schedule interval.
            next_target = self.schedule[self._scheduled_idx]
            remaining = next_target - self.time
            # Clip to avoid overshooting (bug in super).
            if self.dt > remaining > 0:
                self.dt = remaining
        return self.dt
