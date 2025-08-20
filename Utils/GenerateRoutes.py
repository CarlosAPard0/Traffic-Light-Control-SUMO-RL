import argparse
import os
import sys
import random
import xml.etree.ElementTree as ET
from xml.etree.ElementTree import Element, SubElement, tostring
from xml.dom import minidom
from collections import defaultdict


def parse_network_connections(net_file):
    """Parsea el archivo .net.xml para extraer las conexiones reales entre calles"""
    tree = ET.parse(net_file)
    root = tree.getroot()

    connections = defaultdict(list)
    edges = set()

    for edge in root.findall('edge'):
        edge_id = edge.get('id')
        if not edge_id.startswith(':'):
            edges.add(edge_id)

    for connection in root.findall(".//connection[@from]"):
        from_edge = connection.get('from')
        to_edge = connection.get('to')

        if (from_edge and to_edge and
            not from_edge.startswith(':') and
            not to_edge.startswith(':')):
            connections[from_edge].append(to_edge)

    return dict(connections), list(edges)


def generate_valid_routes(connections, edges, max_routes=100, max_length=7):
    """Genera rutas válidas usando las conexiones reales de la red"""
    valid_routes = []

    for start_edge in edges:
        for _ in range(max_routes // len(edges) + 1):
            route = [start_edge]
            current_edge = start_edge

            for _ in range(random.randint(1, max_length - 1)):
                if current_edge in connections and connections[current_edge]:
                    next_edge = random.choice(connections[current_edge])
                    route.append(next_edge)
                    current_edge = next_edge
                else:
                    break

            if len(route) >= 2:
                valid_routes.append(' '.join(route))

    valid_routes = list(set(valid_routes))[:max_routes]
    return valid_routes


def generate_routes(net_file, phases, output_file):
    try:
        connections, edges = parse_network_connections(net_file)
        print(f"Encontradas {len(edges)} calles y {sum(len(v) for v in connections.values())} conexiones")
        if not edges or not connections:
            print("Error: No se encontraron calles o conexiones válidas en la red")
            return
    except Exception as e:
        print(f"Error al parsear el archivo de red {net_file}: {e}")
        return

    valid_routes = generate_valid_routes(connections, edges)
    if not valid_routes:
        print("Error: No se pudieron generar rutas válidas")
        return

    print(f"Generadas {len(valid_routes)} rutas válidas")

    vehicle_types = [
        {"id": "car", "accel": "2.6", "decel": "4.5", "sigma": "0.5", "length": "5", "maxSpeed": "70"},
        {"id": "bus", "accel": "1.2", "decel": "4.0", "sigma": "0.5", "length": "12", "maxSpeed": "50"},
        {"id": "truck", "accel": "1.0", "decel": "4.0", "sigma": "0.5", "length": "8", "maxSpeed": "60"}
    ]

    # Probabilidades de selección (70% car, 20% bus, 10% truck)
    vehicle_type_ids = [vt["id"] for vt in vehicle_types]
    vehicle_probs = [0.7, 0.2, 0.1]

    routes = Element('routes')
    for vtype in vehicle_types:
        vtype_elem = SubElement(routes, 'vType')
        for attr, value in vtype.items():
            vtype_elem.set(attr, value)

    vehicle_id = 0
    current_time = 0

    for duration, vph in phases:
        interval = 3600 / vph  # segundos entre vehículos
        end_time = current_time + duration

        while current_time < end_time:
            vehicle = SubElement(routes, 'vehicle')
            vehicle.set('id', str(vehicle_id))
            vehicle.set('type', random.choices(vehicle_type_ids, weights=vehicle_probs, k=1)[0])
            vehicle.set('depart', f"{current_time:.2f}")

            route = SubElement(vehicle, 'route')
            route.set('edges', random.choice(valid_routes))

            vehicle_id += 1
            current_time += interval

    rough_string = tostring(routes, 'unicode')
    reparsed = minidom.parseString(rough_string)

    os.makedirs(os.path.dirname(output_file), exist_ok=True)

    with open(output_file, 'w') as f:
        f.write(reparsed.toprettyxml(indent="    "))

    print(f"Archivo de rutas generado: {output_file} con {vehicle_id} vehículos")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Generador de rutas válidas para SUMO')

    default_net = os.path.join("Data", "network.net.xml")
    parser.add_argument('net_file', nargs='?', default=default_net,
                        help='Archivo de red (.net.xml)')

    # Definir fases según lo que pediste (9 minutos en 3 intervalos)
    # 180s a 1800 veh/h (≈1 veh cada 2s)
    # 180s a 3600 veh/h (≈1 veh cada 1s)
    # 180s a 7200 veh/h (≈1 veh cada 0.5s)
    default_phases = "180:1800,180:3600,180:7200"
    parser.add_argument('--phases', default=default_phases,
                        help='Fases en formato tiempo:veh/hr,...')

    # Salida en Data/
    default_out = os.path.join("Data", "routes.rou.xml")
    parser.add_argument('--output', default=default_out,
                        help='Archivo de salida')

    args = parser.parse_args()

    try:
        phases = [(int(p.split(':')[0]), int(p.split(':')[1]))
                  for p in args.phases.split(',')]
    except Exception as e:
        print(f"Error al interpretar las fases: {e}")
        sys.exit(1)

    generate_routes(args.net_file, phases, args.output)
