#include <iostream>
#include <vector>
#include <cmath>
#include <map>
#include <queue>
#include <algorithm>

const double EPS = 1e-9;

struct Pt {
    double x, y;
    bool operator<(const Pt& o) const {
        if (std::fabs(x - o.x) > EPS) return x < o.x;
        if (std::fabs(y - o.y) > EPS) return y < o.y;
        return false;
    }
};

struct Seg { Pt p1, p2; };

struct Star {
    Pt center;
    double max_arm_sq = 0;
    std::vector<int> seg_indices;
};

double distSq(Pt p1, Pt p2) {
    return (p1.x - p2.x) * (p1.x - p2.x) + (p1.y - p2.y) * (p1.y - p2.y);
}

bool onSeg(Pt p, const Seg& s) {
    if (p.x < std::min(s.p1.x, s.p2.x) - EPS || p.x > std::max(s.p1.x, s.p2.x) + EPS ||
        p.y < std::min(s.p1.y, s.p2.y) - EPS || p.y > std::max(s.p1.y, s.p2.y) + EPS) {
        return false;
    }
    double cp = (p.y - s.p1.y) * (s.p2.x - s.p1.x) - (p.x - s.p1.x) * (s.p2.y - s.p1.y);
    return std::fabs(cp) < EPS;
}

std::pair<bool, Pt> getIntersect(const Seg& s1, const Seg& s2) {
    double a1 = s1.p2.y - s1.p1.y, b1 = s1.p1.x - s1.p2.x, c1 = a1 * s1.p1.x + b1 * s1.p1.y;
    double a2 = s2.p2.y - s2.p1.y, b2 = s2.p1.x - s2.p2.x, c2 = a2 * s2.p1.x + b2 * s2.p1.y;
    double det = a1 * b2 - a2 * b1;
    if (std::fabs(det) < EPS) return {false, {0, 0}};
    Pt p = {(b2 * c1 - b1 * c2) / det, (a1 * c2 - a2 * c1) / det};
    if (onSeg(p, s1) && onSeg(p, s2)) return {true, p};
    return {false, {0, 0}};
}

bool isAligned(Pt p1, Pt p2) {
    double dx = p2.x - p1.x;
    double dy = p2.y - p1.y;
    if (std::fabs(dx) < EPS && std::fabs(dy) < EPS) return false;
    return std::fabs(dx) < EPS || std::fabs(dy) < EPS || std::fabs(std::fabs(dx) - std::fabs(dy)) < EPS;
}

int main() {
    std::ios_base::sync_with_stdio(false);
    std::cin.tie(NULL);

    int n;
    std::cin >> n;
    std::vector<Seg> segs(n);
    for (int i = 0; i < n; ++i) {
        std::cin >> segs[i].p1.x >> segs[i].p1.y >> segs[i].p2.x >> segs[i].p2.y;
    }

    Pt src, dst;
    std::cin >> src.x >> src.y;
    std::cin >> dst.x >> dst.y;

    std::vector<Star> stars;
    std::vector<bool> assigned(n, false);
    
    for (int i = 0; i < n; ++i) {
        if (assigned[i]) continue;
        std::map<Pt, int> counts;
        Pt center = {-1, -1};
        int max_c = 0;
        for (int j = i + 1; j < n; ++j) {
            auto res = getIntersect(segs[i], segs[j]);
            if (res.first) {
                counts[res.second]++;
                if (counts[res.second] > max_c) {
                    max_c = counts[res.second];
                    center = res.second;
                }
            }
        }
        
        if (max_c > 0) {
            Star s;
            s.center = center;
            for (int k = 0; k < n; ++k) {
                if (!assigned[k] && onSeg(center, segs[k])) {
                    s.seg_indices.push_back(k);
                    s.max_arm_sq = std::max(s.max_arm_sq, distSq(center, segs[k].p1));
                    s.max_arm_sq = std::max(s.max_arm_sq, distSq(center, segs[k].p2));
                }
            }
            for(int idx : s.seg_indices) assigned[idx] = true;
            stars.push_back(s);
        }
    }
    
    int k = stars.size();
    std::vector<int> start_nodes, end_nodes;

    for (int i = 0; i < k; ++i) {
        for (int seg_idx : stars[i].seg_indices) {
            if (onSeg(src, segs[seg_idx])) start_nodes.push_back(i);
            if (onSeg(dst, segs[seg_idx])) end_nodes.push_back(i);
        }
    }
    if (start_nodes.empty()) {
        for (int i = 0; i < k; ++i) if (isAligned(src, stars[i].center)) start_nodes.push_back(i);
    }
    if (end_nodes.empty()) {
        for (int i = 0; i < k; ++i) if (isAligned(dst, stars[i].center)) end_nodes.push_back(i);
    }
    
    std::sort(start_nodes.begin(), start_nodes.end());
    start_nodes.erase(std::unique(start_nodes.begin(), start_nodes.end()), start_nodes.end());
    std::sort(end_nodes.begin(), end_nodes.end());
    end_nodes.erase(std::unique(end_nodes.begin(), end_nodes.end()), end_nodes.end());

    if (start_nodes.empty() || end_nodes.empty()) {
        std::cout << "Impossible";
        return 0;
    }
    
    std::vector<std::vector<int>> adj(k);
    for (int i = 0; i < k; ++i) {
        for (int j = i + 1; j < k; ++j) {
            double d_sq = distSq(stars[i].center, stars[j].center);
            double arm_sum = sqrt(stars[i].max_arm_sq) + sqrt(stars[j].max_arm_sq);
            if (d_sq <= arm_sum * arm_sum + EPS) {
                adj[i].push_back(j);
                adj[j].push_back(i);
            }
        }
    }
    
    std::queue<std::pair<int, int>> q;
    std::vector<int> dist(k, -1);
    for (int s : start_nodes) { q.push({s, 1}); dist[s] = 1; }
    
    while (!q.empty()) {
        auto [u, d] = q.front();
        q.pop();
        for (int e : end_nodes) {
            if (u == e) {
                std::cout << d;
                return 0;
            }
        }
        for (int v : adj[u]) {
            if (dist[v] == -1) {
                dist[v] = d + 1;
                q.push({v, d + 1});
            }
        }
    }

    std::cout << "Impossible";
    return 0;
}