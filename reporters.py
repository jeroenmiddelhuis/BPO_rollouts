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
        if event_lifecycle == "<start_event>":
            self.logfile.write(str(case_id) + self.sep)
            self.logfile.write(event_name + self.sep)
            self.logfile.write(self.sep)
            self.logfile.write(self.displace(time).strftime(self.time_format) + self.sep)
            self.logfile.write(self.displace(time).strftime(self.time_format))
            self.logfile.write("\n")
            self.logfile.flush()
        elif event_lifecycle == "<task:start>":
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


class ProcessReporter:
    """
    A reporter that heavily depends on the process prototypes (task, start_event, intermediate_event, end_event) to report on what happens.
    It assumes tasks are executed for cases that arrive via start_events and complete at end_events. It measures:
    - nr_started: the number of cases that started.
    - nr_completed: the number of cases that completed.
    - total_wait_time: the sum of waiting times of completed cases.
    - total_proc_time: the sum of processing times of completed cases.
    - total_cycle_time: the sum of cycle times of completed cases.
    - resource_busy_times: a mapping of resource_id -> the time the resource was busy during simulation.
    """

    def __init__(self, warmup_time=0):
        self.resource_busy_times = dict()  # mapping of resource_id -> the time the resource was busy during simulation.
        self.nr_started = 0  # number of cases that started
        self.nr_completed = 0  # number of cases that completed
        self.total_wait_time = 0  # sum of waiting times of completed cases
        self.total_proc_time = 0  # sum of processing times of completed cases
        self.total_cycle_time = 0  # sum of cycle times of completed cases

        self.__status = dict()  # case_id -> (nr_busy_tasks, arrival_time, sum_wait_times, sum_proc_times, time_last_busy_change); time_last_busy_change is the time nr_busy_tasks last went from 0 to 1 or from 1 to 0
        self.__resource_start_times = dict()  # resource -> time
        self.__last_time = 0

        self.warmup_time = warmup_time

    def callback(self, case_id, event_name, event_lifecycle, time, resource=""):
        self.__last_time = time
        if event_lifecycle == "<start_event>":
            self.__status[case_id] = (0, time, 0, 0, time)
            if time >= self.warmup_time:
                self.nr_started += 1
        elif event_lifecycle == "<task:start>":
            self.__resource_start_times[resource] = time
            (nr_busy_tasks, arrival_time, sum_wait_times, sum_proc_times, time_last_busy_change) = self.__status[case_id]
            if nr_busy_tasks == 0:
                sum_wait_times += time - time_last_busy_change
                time_last_busy_change = time
            nr_busy_tasks += 1
            self.__status[case_id] = (nr_busy_tasks, arrival_time, sum_wait_times, sum_proc_times, time_last_busy_change)
        elif event_lifecycle == "<task:complete>":
            if time >= self.warmup_time:
                if resource not in self.resource_busy_times.keys():
                    self.resource_busy_times[resource] = 0
                if self.__resource_start_times[resource] < self.warmup_time:
                    self.resource_busy_times[resource] += time - self.warmup_time
                else:
                    self.resource_busy_times[resource] += time - self.__resource_start_times[resource]
            del self.__resource_start_times[resource]
            (nr_busy_tasks, arrival_time, sum_wait_times, sum_proc_times, time_last_busy_change) = self.__status[case_id]
            if nr_busy_tasks == 1:
                sum_proc_times += time - time_last_busy_change
                time_last_busy_change = time
            nr_busy_tasks -= 1
            self.__status[case_id] = (nr_busy_tasks, arrival_time, sum_wait_times, sum_proc_times, time_last_busy_change)
        elif event_lifecycle == "<end_event>":
            (nr_busy_tasks, arrival_time, sum_wait_times, sum_proc_times, time_last_busy_change) = self.__status[case_id]
            del self.__status[case_id]
            if arrival_time >= self.warmup_time:
                self.nr_completed += 1
                self.total_wait_time += sum_wait_times
                self.total_proc_time += sum_proc_times
                self.total_cycle_time += time - arrival_time

    def print_result(self):
        print("Nr. cases started:            ", self.nr_started)
        print("Nr. cases completed:          ", self.nr_completed)
        print("Avg. waiting time per case:   ", round(self.total_wait_time/self.nr_completed, 3))
        print("Avg. processing time per case:", round(self.total_proc_time/self.nr_completed, 3))
        print("Avg. cycle time per case:     ", round(self.total_cycle_time/self.nr_completed, 3))

        # process the resources that are currently busy
        for resource_id in self.__resource_start_times.keys():
            self.resource_busy_times[resource_id] += self.__last_time - self.__resource_start_times[resource_id]

        self.__resource_start_times.clear()
        
        for resource_id in self.resource_busy_times.keys():
            print("Resource", resource_id, "utilization:", round(self.resource_busy_times[resource_id]/(self.__last_time - self.warmup_time), 2))
    
    def close(self):
        pass