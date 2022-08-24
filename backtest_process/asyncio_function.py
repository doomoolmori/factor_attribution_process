import asyncio


def async_start_end(**kwargs):
    if type(kwargs['number']) == int:
        start = int(len(kwargs['space_set']) / 4) * (kwargs['number'] - 1)
        end = int(len(kwargs['space_set']) / 4) * (kwargs['number'])
        if kwargs['number'] == -1:
            end = len(kwargs['space_set'])
    else:
        start = 0
        end = len(kwargs['space_set'])
    return {'start': start, 'end': end}


async def async_loop(number: int, **kwargs):
    kwargs['input']['number'] = number
    loop = asyncio.get_event_loop()
    await loop.run_in_executor(
        None,
        kwargs['function'],
        kwargs['input'])


async def async_main(**kwargs):
    await asyncio.gather(
        async_loop(number=1, **kwargs),
        async_loop(number=2, **kwargs),
        async_loop(number=3, **kwargs),
        async_loop(number=-1, **kwargs))
