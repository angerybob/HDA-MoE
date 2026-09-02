"""Topology abstractions for HDA-MoE communication simulation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


class Topology:
    def num_endpoints(self) -> int:
        raise NotImplementedError

    def get_route(self, src: int, dst: int, flow_id: Any = None) -> list[Any]:
        routes = self.get_candidate_routes(src, dst)
        if not routes:
            return []
        if flow_id is None:
            return routes[0]
        try:
            idx = int(flow_id) % len(routes)
        except Exception:
            idx = hash(flow_id) % len(routes)
        return routes[idx]

    def get_candidate_routes(self, src: int, dst: int) -> list[list[Any]]:
        raise NotImplementedError

    def link_bandwidth(self, link_id: Any) -> float:
        raise NotImplementedError

    def endpoint_group(self, endpoint: int) -> dict[str, int]:
        return {"endpoint": int(endpoint)}


@dataclass
class MeshTopology(Topology):
    x: int
    y: int
    link_bw: float

    def num_endpoints(self) -> int:
        return self.x * self.y

    def _coord(self, endpoint: int) -> tuple[int, int]:
        return int(endpoint) // self.y, int(endpoint) % self.y

    def _endpoint(self, row: int, col: int) -> int:
        return int(row) * self.y + int(col)

    def get_candidate_routes(self, src: int, dst: int) -> list[list[Any]]:
        src_coord = self._coord(src)
        dst_coord = self._coord(dst)
        current = src_coord
        path = []
        while current[0] != dst_coord[0]:
            nxt = (current[0] + (1 if dst_coord[0] > current[0] else -1), current[1])
            path.append((self._endpoint(*current), self._endpoint(*nxt)))
            current = nxt
        while current[1] != dst_coord[1]:
            nxt = (current[0], current[1] + (1 if dst_coord[1] > current[1] else -1))
            path.append((self._endpoint(*current), self._endpoint(*nxt)))
            current = nxt
        return [path]

    def link_bandwidth(self, link_id: Any) -> float:
        return self.link_bw

    def endpoint_group(self, endpoint: int) -> dict[str, int]:
        r, c = self._coord(endpoint)
        return {"endpoint": int(endpoint), "row": r, "col": c}


@dataclass
class TorusTopology(MeshTopology):
    def _axis_steps(self, cur: int, dst: int, size: int) -> list[int]:
        forward = (dst - cur) % size
        backward = (cur - dst) % size
        if forward <= backward:
            return [1] * forward
        return [-1] * backward

    def get_candidate_routes(self, src: int, dst: int) -> list[list[Any]]:
        src_coord = self._coord(src)
        dst_coord = self._coord(dst)
        current = src_coord
        path = []
        for step in self._axis_steps(current[0], dst_coord[0], self.x):
            nxt = ((current[0] + step) % self.x, current[1])
            path.append((self._endpoint(*current), self._endpoint(*nxt)))
            current = nxt
        for step in self._axis_steps(current[1], dst_coord[1], self.y):
            nxt = (current[0], (current[1] + step) % self.y)
            path.append((self._endpoint(*current), self._endpoint(*nxt)))
            current = nxt
        return [path]


class FatTreeTopology(Topology):
    def __init__(
        self,
        num_pods: int,
        leaf_per_pod: int,
        endpoints_per_leaf: int,
        agg_per_pod: int,
        core_count: int,
        endpoint_bw: float,
        leaf_agg_bw: float,
        agg_core_bw: float,
        oversubscription: float = 1.0,
        routing: str = "ecmp_rr",
    ) -> None:
        self.num_pods = int(num_pods)
        self.leaf_per_pod = int(leaf_per_pod)
        self.endpoints_per_leaf = int(endpoints_per_leaf)
        self.agg_per_pod = int(agg_per_pod)
        self.core_count = int(core_count)
        self.endpoint_bw = float(endpoint_bw)
        self.leaf_agg_bw = float(leaf_agg_bw)
        self.agg_core_bw = float(agg_core_bw) / float(oversubscription)
        self.oversubscription = float(oversubscription)
        self.routing = routing
        self._n = self.num_pods * self.leaf_per_pod * self.endpoints_per_leaf

    def num_endpoints(self) -> int:
        return self._n

    def _ids(self, endpoint: int) -> tuple[int, int, int]:
        endpoint = int(endpoint)
        leaf_global = endpoint // self.endpoints_per_leaf
        pod = leaf_global // self.leaf_per_pod
        leaf = leaf_global % self.leaf_per_pod
        offset = endpoint % self.endpoints_per_leaf
        return pod, leaf, offset

    def endpoint_group(self, endpoint: int) -> dict[str, int]:
        pod, leaf, offset = self._ids(endpoint)
        return {"endpoint": int(endpoint), "pod": pod, "leaf": leaf, "endpoint_in_leaf": offset}

    def _leaf_id(self, pod: int, leaf: int) -> tuple[str, int, int]:
        return ("leaf", int(pod), int(leaf))

    def _agg_id(self, pod: int, agg: int) -> tuple[str, int, int]:
        return ("agg", int(pod), int(agg))

    def _core_id(self, core: int) -> tuple[str, int]:
        return ("core", int(core))

    def get_candidate_routes(self, src: int, dst: int) -> list[list[Any]]:
        if int(src) == int(dst):
            return []
        sp, sl, _ = self._ids(src)
        dp, dl, _ = self._ids(dst)
        src_leaf = self._leaf_id(sp, sl)
        dst_leaf = self._leaf_id(dp, dl)
        routes: list[list[Any]] = []

        if sp == dp and sl == dl:
            routes.append([
                (("endpoint", int(src)), src_leaf),
                (src_leaf, ("endpoint", int(dst))),
            ])
            return routes

        if sp == dp:
            for agg in range(self.agg_per_pod):
                agg_id = self._agg_id(sp, agg)
                routes.append([
                    (("endpoint", int(src)), src_leaf),
                    (src_leaf, agg_id),
                    (agg_id, dst_leaf),
                    (dst_leaf, ("endpoint", int(dst))),
                ])
            return routes

        for sagg in range(self.agg_per_pod):
            for core in range(self.core_count):
                dagg = core % self.agg_per_pod
                sagg_id = self._agg_id(sp, sagg)
                dagg_id = self._agg_id(dp, dagg)
                core_id = self._core_id(core)
                routes.append([
                    (("endpoint", int(src)), src_leaf),
                    (src_leaf, sagg_id),
                    (sagg_id, core_id),
                    (core_id, dagg_id),
                    (dagg_id, dst_leaf),
                    (dst_leaf, ("endpoint", int(dst))),
                ])
        return routes

    def link_bandwidth(self, link_id: Any) -> float:
        a, b = link_id
        kinds = {a[0], b[0]}
        if "endpoint" in kinds:
            return self.endpoint_bw
        if kinds == {"leaf", "agg"}:
            return self.leaf_agg_bw
        if kinds == {"agg", "core"}:
            return self.agg_core_bw
        return self.endpoint_bw


def build_topology(topology_type: str, topology_config: dict[str, Any] | None, D: int, BW: float, x: int, y: int) -> Topology:
    cfg = dict(topology_config or {})
    topo = topology_type.lower()
    if topo == "mesh":
        return MeshTopology(x=x, y=y, link_bw=BW)
    if topo == "torus":
        return TorusTopology(x=x, y=y, link_bw=BW)
    if topo in {"fat_tree", "fattree"}:
        cfg.setdefault("num_pods", 4)
        cfg.setdefault("leaf_per_pod", 4)
        cfg.setdefault("endpoints_per_leaf", 2)
        cfg.setdefault("agg_per_pod", 2)
        cfg.setdefault("core_count", 4)
        cfg.setdefault("endpoint_bw", BW)
        cfg.setdefault("leaf_agg_bw", BW)
        cfg.setdefault("agg_core_bw", BW)
        cfg.setdefault("oversubscription", 1.0)
        ft = FatTreeTopology(**cfg)
        if ft.num_endpoints() != D:
            raise ValueError(f"Fat-tree endpoints {ft.num_endpoints()} != D={D}")
        return ft
    raise ValueError(f"Unknown topology_type={topology_type}")
