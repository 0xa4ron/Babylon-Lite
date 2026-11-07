import { describe, expect, it } from "vitest";

import { composeMat4 } from "../../../packages/babylon-lite/src/math/compose-mat4";
import type { Mat4 } from "../../../packages/babylon-lite/src/math/types";
import type { Mesh } from "../../../packages/babylon-lite/src/mesh/mesh";
import { createEmptyPickingInfo, type PickingInfo } from "../../../packages/babylon-lite/src/picking/picking-info";
import { getPickedNormal } from "../../../packages/babylon-lite/src/picking/picking-helpers";

function fakeGpuHit(worldMatrix: Mat4, normal: [number, number, number] = [1, 0, 1]): PickingInfo {
    const mesh = {
        worldMatrix,
        _cpuIndices: new Uint32Array([0, 1, 2]),
        _cpuNormals: new Float32Array([...normal, ...normal, ...normal]),
    } as Mesh;
    const info = createEmptyPickingInfo();
    info.hit = true;
    info.pickedMesh = mesh;
    info.faceId = 0;
    info.bu = 1;
    info.bv = 0;
    // Unlike the precise CPU picker, GPU-picked faces can provide barycentrics
    // without a precomputed pickedNormalWorld.
    return info;
}

function scale(x: number, y: number, z: number): Mat4 {
    return composeMat4(0, 0, 0, 0, 0, 0, 1, x, y, z);
}

describe("getPickedNormal for GPU-picked barycentric fallback", () => {
    it("uses inverse-transpose under non-uniform scale rather than forward world scaling", () => {
        const hit = fakeGpuHit(scale(2, 1, 1));
        const world = getPickedNormal(hit, true)!;
        expect(world[0]).toBeCloseTo(1 / Math.sqrt(5), 6);
        expect(world[1]).toBeCloseTo(0, 6);
        expect(world[2]).toBeCloseTo(2 / Math.sqrt(5), 6);
    });

    it("faces the returned local normal according to the correctly transformed world normal", () => {
        const hit = fakeGpuHit(scale(2, 1, 1));
        hit.ray = { origin: [0, 0, -2], direction: [0, 0, 1], length: 10 };
        const local = getPickedNormal(hit, false)!;
        expect(local[0]).toBeCloseTo(-Math.SQRT1_2, 6);
        expect(local[1]).toBeCloseTo(0, 6);
        expect(local[2]).toBeCloseTo(-Math.SQRT1_2, 6);
        const world = getPickedNormal(hit, true)!;
        expect(world[0]).toBeCloseTo(-1 / Math.sqrt(5), 6);
        expect(world[2]).toBeCloseTo(-2 / Math.sqrt(5), 6);
    });

    it("handles a sheared transform whose normal direction differs from the vertex direction", () => {
        const sheared = new Float32Array([1, 0, 0, 0, 1, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1]) as unknown as Mat4;
        const hit = fakeGpuHit(sheared, [1, 0, 0]);
        const world = getPickedNormal(hit, true)!;
        expect(world[0]).toBeCloseTo(Math.SQRT1_2, 6);
        expect(world[1]).toBeCloseTo(-Math.SQRT1_2, 6);
        expect(world[2]).toBeCloseTo(0, 6);
    });

    it("includes the selected thin-instance matrix and mirrored handedness", () => {
        const hit = fakeGpuHit(scale(2, 1, 1));
        const matrices = new Float32Array(32);
        matrices.set(scale(1, 1, 1) as unknown as Float32Array, 0);
        matrices.set(scale(-1, 1, 0.5) as unknown as Float32Array, 16);
        (hit.pickedMesh as Mesh).thinInstances = { matrices, count: 2 } as NonNullable<Mesh["thinInstances"]>;
        hit.thinInstanceIndex = 1;

        const world = getPickedNormal(hit, true)!;
        const len = Math.hypot(0.5, 2);
        expect(world[0]).toBeCloseTo(-0.5 / len, 6);
        expect(world[1]).toBeCloseTo(0, 6);
        expect(world[2]).toBeCloseTo(2 / len, 6);
        // Neither the original mesh nor its instance data was modified.
        expect(matrices[16]).toBe(-1);
    });

    it("rejects missing instances and singular normal transforms", () => {
        const hit = fakeGpuHit(scale(1, 1, 1));
        hit.thinInstanceIndex = 0;
        expect(getPickedNormal(hit, true)).toBeNull();
        const singular = fakeGpuHit(scale(0, 1, 1));
        expect(getPickedNormal(singular, true)).toBeNull();
    });
});
