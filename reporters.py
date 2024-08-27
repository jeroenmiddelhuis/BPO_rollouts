from datetime import datetime, timedelta
from enum import Enum, auto


class TimeUnit(Enum):
    """An enumeration for the unit in which simulation time is measured."""
    SECONDS = auto()
    """
    Measured in seconds.
    
    :meta hide-value:
    """
    MINUTES = auto()
    """
    Measured in minutes.

    :meta hide-value:
    """
    HOURS = auto()
    """
    Measured in hours.

    :meta hide-value:
    """
    DAYS = auto()
    """
    Measured in days.

    :meta hide-value:
    """


class EventLogReporter:
    """
    :param filename: the name of the file in which the event log must be stored.
    :param timeunit: the :class:`.TimeUnit` of simulation time.
    :param initial_time: a datetime value.
    :param time_format: a datetime formatting string.
    :param data_fields: the data fields to report in the log.
    :param separator: the separator to use in the log.
    """
    def __init__(self, filename, timeunit=TimeUnit.MINUTES, initial_time=datetime(2020, 1, 1), time_format="%Y-%m-%d %H:%M:%S.%f", separator=","):
        self.task_start_times = dict()
        self.timeunit = timeunit
        self.initial_time = initial_time
        self.time_format = time_format
        self.logfile = open(filename, "wt")
        self.sep = separator        
        self.logfile.write("case_id"+self.sep+"task"+self.sep+"resource"+self.sep+"start_time"+self.sep+"completion_time\n")

    def displace(self, time):
        return self.initial_time + (timedelta(seconds=time) if self.timeunit == TimeUnit.SECONDS else timedelta(
            minutes=time) if self.timeunit == TimeUnit.MINUTES else timedelta(
            hours=time) if self.timeunit == TimeUnit.HOURS else timedelta(
            days=time) if self.timeunit == TimeUnit.DAYS else None)

    def callback(self, case_id, event_name, event_lifecycle, time, resource=""):
        if event_lifecycle == "<task:start>":
            self.task_start_times[(case_id, event_name)] = time
        elif event_lifecycle == "<task:complete>":
            if (case_id, event_name) in self.task_start_times.keys():
                self.logfile.write(str(case_id) + self.sep)
                self.logfile.write(event_name + self.sep)
                self.logfile.write(str(resource) + self.sep)
                self.logfile.write(self.displace(self.task_start_times[(case_id, event_name)]).strftime(self.time_format) + self.sep)
                self.logfile.write(self.displace(time).strftime(self.time_format))
                self.logfile.write("\n")
                self.logfile.flush()
                del self.task_start_times[(case_id, event_name)]
        elif event_lifecycle == "<event:complete>":
            self.logfile.write(str(case_id) + self.sep)
            self.logfile.write(event_name + self.sep)
            self.logfile.write(self.sep)
            self.logfile.write(self.displace(time).strftime(self.time_format) + self.sep)
            self.logfile.write(self.displace(time).strftime(self.time_format))
            self.logfile.write("\n")
            self.logfile.flush()

    def close(self):
        self.logfile.close()