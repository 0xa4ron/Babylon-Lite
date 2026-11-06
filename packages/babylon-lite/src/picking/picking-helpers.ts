import type { PickingInfo } from "./picking-info.js";
import type { Mesh } from "../mesh/mesh.js";
import { normalizeVec3TupleOrUp } from "../math/normalize-vec3-tuple-or-up.js";
import { invertMat4 } from "../math/invert-mat4.js";
import { multiplyMat4 } from "../math/multiply-mat4.js";
import type { Mat4 } from "../math/types.js";

/**
 * Get the interpolated normal at the picked point.
 * Requires detailed picking (`faceId >= 0`) and mesh._cpuNormals.
 * @param useWorldCoordinates - if true, transform normal by world matrix (default: false)
 */
export function getPickedNormal(info: PickingInfo, useWorldCoordinates = false): [number, number, number] | null {
    if (info._normalsInvalid) {
        return null;
    }
    if (useWorldCoordinates && info.pickedNormalWorld) {
        return info.pickedNormalWorld;
    }
    if (!useWorldCoordinates && info.pickedNormal) {
        return info.pickedNormal;
    }

    const mi = info.pickedMesh as Mesh | undefined;
    if (info.faceId < 0 || !mi || !mi._cpuNormals || !mi._cpuIndices) {
        return null;
    }

    const normals = mi._cpuNormals;
    const indices = mi._cpuIndices;
    const face = info.faceId;

    const i0 = indices[face * 3]!;
    const i1 = indices[face * 3 + 1]!;
    const i2 = indices[face * 3 + 2]!;

    // BJS exposes bu for vertex 0 and bv for vertex 1; vertex 2 gets the remainder.
    const bw = 1 - info.bu - info.bv;
    const nx = info.bu * normals[i0 * 3]! + info.bv * normals[i1 * 3]! + bw * normals[i2 * 3]!;
    const ny = info.bu * normals[i0 * 3 + 1]! + info.bv * normals[i1 * 3 + 1]! + bw * normals[i2 * 3 + 1]!;
    const nz = info.bu * normals[i0 * 3 + 2]! + info.bv * normals[i1 * 3 + 2]! + bw * normals[i2 * 3 + 2]!;

    const localNormal = normalizeVec3TupleOrUp(nx, ny, nz);
    // GPU-picked faces may have barycentrics but no cached surface normal.
    // Transform their CPU-interpolated normal with the inverse-transpose,
    // not the world 3x3 (which is wrong under non-uniform scale or shear).
    let world = mi.worldMatrix;
    if (info.thinInstanceIndex >= 0) {
        const matrices = mi.thinInstances?.matrices;
        const offset = info.thinInstanceIndex * 16;
        if (!matrices || !Number.isInteger(offset) || offset + 16 > matrices.length) {
            return null;
        }
        // The instance transform is local to the mesh: world * instance.
        // Copy rather than mutate the caller's shared thin-instance buffer.
        const instance = new Float32Array(matrices.subarray(offset, offset + 16));
        instance[3] = instance[7] = instance[11] = 0;
        instance[15] = 1;
        world = multiplyMat4(world, instance as unknown as Mat4);
    }
    const inverse = invertMat4(world);
    if (!inverse) {
        return null; // A singular world transform has no defined world-space surface normal.
    }
    const wnx = inverse[0]! * localNormal[0] + inverse[1]! * localNormal[1] + inverse[2]! * localNormal[2];
    const wny = inverse[4]! * localNormal[0] + inverse[5]! * localNormal[1] + inverse[6]! * localNormal[2];
    const wnz = inverse[8]! * localNormal[0] + inverse[9]! * localNormal[1] + inverse[10]! * localNormal[2];
    const worldNormal = normalizeVec3TupleOrUp(wnx, wny, wnz);
    const flip = info.ray ? worldNormal[0] * info.ray.direction[0] + worldNormal[1] * info.ray.direction[1] + worldNormal[2] * info.ray.direction[2] > 0 : false;

    if (!useWorldCoordinates) {
        return flip ? [-localNormal[0], -localNormal[1], -localNormal[2]] : localNormal;
    }

    return flip ? [-worldNormal[0], -worldNormal[1], -worldNormal[2]] : worldNormal;
}

export function getPickedFaceNormal(info: PickingInfo, useWorldCoordinates = false): [number, number, number] | null {
    if (info._normalsInvalid) {
        return null;
    }
    if (useWorldCoordinates) {
        return info.pickedFaceNormalWorld;
    }
    return info.pickedFaceNormal;
}

/**
 * Get the interpolated UV coordinates at the picked point.
 * Requires detailed picking (`faceId >= 0`) and mesh._cpuUvs.
 */
export function getPickedUV(info: PickingInfo): [number, number] | null {
    const mi = info.pickedMesh as Mesh | undefined;
    if (info.faceId < 0 || !mi || !mi._cpuUvs || !mi._cpuIndices) {
        return null;
    }

    const uvs = mi._cpuUvs;
    const indices = mi._cpuIndices;
    const face = info.faceId;

    const i0 = indices[face * 3]!;
    const i1 = indices[face * 3 + 1]!;
    const i2 = indices[face * 3 + 2]!;

    const bw = 1 - info.bu - info.bv;
    const u = info.bu * uvs[i0 * 2]! + info.bv * uvs[i1 * 2]! + bw * uvs[i2 * 2]!;
    const v = info.bu * uvs[i0 * 2 + 1]! + info.bv * uvs[i1 * 2 + 1]! + bw * uvs[i2 * 2 + 1]!;

    return [u, v];
}
