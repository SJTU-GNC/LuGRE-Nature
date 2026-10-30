"""Optional validation-only guard: reject data reads outside the relocated bundle."""
import json, os, sys
if os.environ.get('LUGRE_PORTABLE_GUARD')=='1':
    from pathlib import Path
    _roots=[Path(p).resolve() for p in json.loads(os.environ['LUGRE_ALLOWED_ROOTS'])]
    _log=Path(os.environ['LUGRE_GUARD_LOG'])
    _busy=False
    def _deny(event,value):
        global _busy
        if _busy:return
        _busy=True
        try:
            with _log.open('a',encoding='utf-8') as f:
                f.write(json.dumps({'event':event,'path':str(value)},ensure_ascii=False)+'\n')
        finally:_busy=False
        raise PermissionError('Portable validation rejected external access: '+str(value))
    def _hook(event,args):
        if _busy:return
        if event in ('socket.connect','socket.getaddrinfo'):_deny(event,args[0] if args else '')
        if event not in ('open','os.listdir','os.scandir'):return
        value=args[0] if args else None
        if value is None or isinstance(value,int):return
        try:path=Path(os.fsdecode(value)).resolve()
        except (ValueError,TypeError,OSError):return
        if not any(path==r or r in path.parents for r in _roots):_deny(event,path)
    sys.addaudithook(_hook)
