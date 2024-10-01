#!/usr/bin/env python3

import time
import radical.pilot as rp


if __name__ == '__main__':

    session = rp.Session()
    try:

        pmgr  = rp.PilotManager(session=session)
        tmgr  = rp.TaskManager(session=session)
        pdesc = rp.PilotDescription({'resource': 'local.localhost',
                                     'runtime' : 60,
                                     'nodes'   : 3})
        pilot = pmgr.submit_pilots(pdesc)
        tmgr.add_pilots(pilot)

        url = 'http://127.0.0.1:11000'
        sd  = rp.TaskDescription(
              {'uid'         : 'ollama',
               'mode'        : rp.TASK_SERVICE,
               'environment' : {'OLLAMA_HOST': url},
               'executable'  : '/project/bii_dsc_community/djy8hg/arupcsedu/cylonplus/src/radical-dl/ollama',
               'arguments'   : ['start'],
               'named_env'   : 'rp'})

        service = tmgr.submit_tasks(sd)
        tmgr.wait_tasks(uids=[service.uid], state=[rp.AGENT_EXECUTING])

        n   = 2
        tds = list()
        for _ in range(n):
            td  = rp.TaskDescription({'executable' : '/project/bii_dsc_community/djy8hg/arupcsedu/cylonplus/src/radical-dl/ollama_client.py',
                                      'ranks'      : 1,
                                      'environment': {'OLLAMA_URL': url},
                                      'named_env'  : 'rp'})
            tds.append(td)

        tasks = tmgr.submit_tasks(tds)

        tmgr.wait_tasks(uids=[t.uid for t in tasks])

        for task in tasks:
            print('---- %s' % task.uid)
            print('STDERR: %s' % task.stderr)
            print('STDOUT: %s' % task.stdout)

    finally:
        session.close(download=True)


# ------------------------------------------------------------------------------
