def loop_with_callback(callback, iterations=10):
    a = 0
    for i in range(iterations):
        a += 1
        callback(a)

# def print_current_value(value):
#     print(f"Current value of a: {value}")
#
# def main():
#     # 调用 loop_with_callback 函数，并传递 print_current_value 作为回调函数
#     loop_with_callback(print_current_value)
#
# if __name__ == "__main__":
#     main()
def collect_values(values_list):
    def add_value(value):
        values_list.append(value)
    return add_value

values = []
loop_with_callback(collect_values(values), 10)

print("Collected values:", values)