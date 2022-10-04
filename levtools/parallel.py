import multiprocessing as mp
import queue
import traceback

mp.freeze_support() # for windows
#mp.set_start_method("spawn")


def parallel_executions_on_same_data(worker_function, args, workers=mp.cpu_count(), debug=False, **kwargs):
    """
    returns: [func(args, **kwargs) for _ in range(workers)]

    executed in workers number of processes
    """

    if debug:
        return [worker_function(args, **kwargs) for _ in range(workers)]

    def wrapped_func(child_pipe_endpoint):
        args = child_pipe_endpoint.recv()
        kwargs = child_pipe_endpoint.recv()
        returned = worker_function(args, **kwargs)
        child_pipe_endpoint.send(returned)

    pipes = [mp.Pipe() for _ in range(workers)]
    processes = [mp.Process(target=wrapped_func, args=(child_end,))
                 for parent_end, child_end in pipes]

    for (parent_end, _) in pipes:
        parent_end.send(args)
        parent_end.send(kwargs)

    for process in processes:
        process.start()

    for process in processes:
        process.join()

    for process in processes:
        process.close()

    return [parent_end.recv() for parent_end, _ in pipes]


def _function_wrapped(worker_function, task):
    try:
        if isinstance(task, dict):
            result = worker_function(**task)
        elif isinstance(task, (tuple, list)):
            result = worker_function(*task)
        else:
            result = worker_function(task)

        return result

    except:
        traceback.print_exc()
        return None


def _func_executor_cycle(worker_function, tasks_queue, results_queue, *args, **kwargs):
    """
    Assumes queue already contains all tasks.
    End of Tasks is signaled by None
    In case of execution error None is put in result_queue
    """
    while True:
        try:
            item = tasks_queue.get_nowait()

            if not item:  # End of queue signal: None
                break

            inx, task = item
            result = _function_wrapped(worker_function, task)
            results_queue.put((inx, result))

        except queue.Empty:
            break

        except:
            traceback.print_exc()


def _func_executor_from_queues(worker_function, task_queue, result_queue):
    inp = task_queue.get()
    ret = _function_wrapped(worker_function, inp)
    result_queue.put(ret)


def clear(q):
    try:
        while True:
            q.get_nowait()
    except queue.Empty:
        pass


def _supervisor(worker_function, tasks_queue, results_queue, timeout):
    down_tasks_queue = mp.Queue()
    down_results_queue = mp.Queue()
    name = mp.current_process().name
    print(f"{name}: Started", flush=True)

    while not tasks_queue.empty():
        inp_inx, inp = tasks_queue.get()
        down_tasks_queue.put(inp)

        proc = mp.Process(name=name + '_executor',
                          target=_func_executor_from_queues,
                          args=(worker_function, down_tasks_queue, down_results_queue))

        proc.start()
        try:
            result = down_results_queue.get(timeout=timeout)
            print(f"{name}: success for {inp}", flush=True)
            results_queue.put((inp_inx, result))
        except queue.Empty:
            proc.kill()
            clear(down_results_queue)
            results_queue.put((inp_inx, None))
        proc.join()

    print(f"{name}: Finished", flush=True)


def map(func, iterable, timeout=None, workers=mp.cpu_count()):
    """
    args: list of tuples. Each tuple represents data to be sent to func
    Input and outputs are not mixed up by execution speed, so they remain sorted.

    main: puts all tasks into a down queue and starts n manager processes.
    Each manager process gets the job from the queue and sends it another sub-process via it's down queue.
    If process completes in timeout then gets return values otherwise kills subprocess and starts another.
    """

    executor = _supervisor if timeout and timeout > 0 else _func_executor_cycle

    n_items = len(iterable)
    tasks_queue = mp.Queue()
    results_queue = mp.Queue()
    processes = [mp.Process(name="manager: " + str(worker_inx),
                            target=executor,
                            args=(func, tasks_queue, results_queue, timeout))
                 for worker_inx in range(workers)]

    for inp_inx, item in enumerate(iterable):
        tasks_queue.put((inp_inx, item))

    for process in processes:
        process.start()

    recv_list = [None] * n_items
    for ix in range(n_items):
        inp_ix, recv = results_queue.get()  # it may block here if one of the processes blocks (non-timeout case)
                                            # or if supervisor hangs
        recv_list[inp_ix] = recv

    for process in processes:
           process.join()

    return recv_list
