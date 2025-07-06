from prody import LOGGER
import time

# Add a _times dict if not already present
if not hasattr(LOGGER, '_times'):
    LOGGER._times = {}

if not hasattr(LOGGER, '_reports'):
    LOGGER._reports = {}

if not hasattr(LOGGER, '_report_times'):
    LOGGER._report_times = {}

# Define your replacement method
def custom_report(self, msg='Completed in %.2fs.', label=None):
    if label not in self._times:
        self.warning(f"No timing info for label '{label}'")
        return
    elapsed = time.time() - self._times[label]
    self.debug(msg % elapsed)

    if label not in self._reports:
        self._reports[label] = elapsed
        self._report_times[label] = 1
    else:
        self._reports[label] += elapsed
        self._report_times[label] += 1

# Monkey-patch: bind the method to LOGGER
import types
LOGGER.report = types.MethodType(custom_report, LOGGER)

