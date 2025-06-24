

def ass():
    only_names = ["Ana", "Mark", "Ass", "Dick"]
    full_names = [{"Mark", "Fenderov"},{"Ana", "Helena", "Maldonado", "Labara"}]

    name_table = set()
    for name in only_names:
        name_table.add(name)
    print(name_table)

    for full_name in full_names:
        for name in full_name:
            if name in name_table:
                print(name+" is a name")
            else:
                print(name+ " is not a name")                


ass()