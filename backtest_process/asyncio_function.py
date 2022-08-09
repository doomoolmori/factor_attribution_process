import asyncio

async def async_loop(number:int, **kwargs):
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

