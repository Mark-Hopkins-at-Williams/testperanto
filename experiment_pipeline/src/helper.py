def format_number(num):
    if num >= 1000000:
        return f"{num/1000000:.1f}m"
    elif num >= 1000:
        return f"{num/1000:.1f}k"
    else:
        return str(num)