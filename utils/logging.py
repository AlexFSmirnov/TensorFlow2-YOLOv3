def format_seconds(s):
    hours, rem = divmod(int(s), 3600)
    minutes, seconds = divmod(rem, 60)
    return f'{hours:02}h {minutes:02}m {seconds:02}s'
