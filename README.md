# parseDomeVRLog
Python module containing the class TextLog to parse the DomeVRLog files produced by the DomeVR project or TimedLog plugin. 

Uses memmap to speed up the parsing of large files.

### Example usage
```
from parse_domevrlog import TextLog
with TextLog(filename) as log:
    start_trial_times = log.parse_all_state_times(state='StartTrial', times='StateStarted')
    end_trial_times = log.parse_all_state_times(state='EndTrial', times='StateStarted')
print(start_trial_times)
print(end_trial_times)
```
