create table distribution (
    type char(128),
    property char(128),
    unit char(218),
    amount REAL,
    primary key (type, property, unit, amount),
    foreign key (type) references relation(type),
    foreign key (property) references relation(property),
    foreign key (unit) references relation(unit)
);


