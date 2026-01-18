import fs from 'node:fs'
import path from 'node:path'
import { describe, expect, it } from 'vitest'

import { fingerprintDiff } from '../index'

describe('diff fingerprint', () => {
  it('generates a same fingerprint for similar diffs', async () => {
    const fixturesDir = path.join(__dirname, './__fixtures__')

    const generateFingerprintFromFilename = async (name: string) => {
      const diffPath = path.join(fixturesDir, name)
      const buffer = fs.readFileSync(diffPath)
      console.time(name)
      const res = await fingerprintDiff(buffer)
      console.timeEnd(name)
      return res
    }

    const generateFingerprintFromPath = async (name: string) => {
      const diffPath = path.join(fixturesDir, name)
      console.time(`${name}-path`)
      const res = await fingerprintDiff(diffPath)
      console.timeEnd(`${name}-path`)
      return res
    }

    const fA1 = await generateFingerprintFromFilename('diff-A1.png')
    const fA1Path = await generateFingerprintFromPath('diff-A1.png')
    const fA2 = await generateFingerprintFromFilename('diff-A2.png')
    const fA3 = await generateFingerprintFromFilename('diff-A3.png')
    const fB1 = await generateFingerprintFromFilename('diff-B1.png')
    const fLong = await generateFingerprintFromFilename('long-diff.png')
    const weirdDiff = await generateFingerprintFromFilename('weird-diff.png')
    const weirdDiff2 = await generateFingerprintFromFilename('weird-diff-2.png')

    expect(weirdDiff).toBe('v1:g16:d1:t0.002,0.02,0.08:badfb9bd5eea0beb')
    expect(weirdDiff2).toBe('v1:g16:d1:t0.002,0.02,0.08:ccb73aa8a5d0317b')

    expect(fA1).toBe('v1:g16:d1:t0.002,0.02,0.08:202566ca9533046b')
    expect(fA1).toBe(fA2)
    expect(fA1).toBe(fA3)
    expect(fA1).toBe(fA1Path)
    expect(fA1).not.toBe(fB1)
    expect(fLong).not.toBe(fB1)
  })
})
