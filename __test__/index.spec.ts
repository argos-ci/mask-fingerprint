import fs from 'node:fs'
import path from 'node:path'
import { describe, expect, it } from 'vitest'

import { fingerprintDiffForEquality } from '../index'

describe('diff fingerprint', () => {
  it('generates a same fingerprint for similar diffs', () => {
    const fixturesDir = path.join(__dirname, './__fixtures__')

    const generateFingerprintFromFilename = (name: string) => {
      const diffPath = path.join(fixturesDir, name)
      const buffer = fs.readFileSync(diffPath)
      console.time(name)
      const res = fingerprintDiffForEquality(buffer)
      console.timeEnd(name)
      return res
    }

    const fA1 = generateFingerprintFromFilename('diff-A1.png')
    const fA2 = generateFingerprintFromFilename('diff-A2.png')
    const fA3 = generateFingerprintFromFilename('diff-A3.png')
    const fB1 = generateFingerprintFromFilename('diff-B1.png')
    const fLong = generateFingerprintFromFilename('long-diff.png')

    expect(fA1).toBe(fA2)
    expect(fA1).toBe(fA3)
    expect(fA1).not.toBe(fB1)
    expect(fLong).not.toBe(fB1)
  })
})
